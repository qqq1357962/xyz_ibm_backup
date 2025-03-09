#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <ATen/cuda/Atomic.cuh>

#include "density_map_cuda.cuh"

template <typename scalar_t>
__device__ scalar_t overlap(scalar_t x_l, scalar_t x_h, scalar_t bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    float *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    atomicAdd(&aux_mat[j * num_bin_y + k], weight * overlap_area);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const float *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            extern __shared__ unsigned char grad_xy[];
            float *grad_x = (float *)grad_xy;
            float *grad_y = grad_x + blockDim.z;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                grad_x[threadIdx.z] = grad_y[threadIdx.z] = 0;
            }
            __syncthreads();

            float part_grad_x = 0;
            float part_grad_y = 0;

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    float tmp_x = grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    float tmp_y = grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    part_grad_x += overlap_area * tmp_x;
                    part_grad_y += overlap_area * tmp_y;
                }
            }
            atomicAdd(&grad_x[threadIdx.z], part_grad_x);
            atomicAdd(&grad_y[threadIdx.z], part_grad_y);
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
                node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
            }
        }
    }
}


//---------------------------------------------------------------------

__global__ void density_map_cuda_forward_naive_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    float *aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float node_w = node_size[i][0];
        float node_h = node_size[i][1];
        float ratio = 1.0;
        if (clamp_node) {
            const float node_area = node_w * node_h;
            node_w = max(node_w, static_cast<float>(min_node_w));
            node_h = max(node_h, static_cast<float>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const float mgn = static_cast<float>(margin);
        const float num_bin_x_minus_mgn = static_cast<float>(num_bin_x) - mgn;
        const float num_bin_y_minus_mgn = static_cast<float>(num_bin_y) - mgn;
        const float small_mgn = static_cast<float>(margin * 0.1);
        float x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        float x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        float y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        float y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }
        const float p_node_wght = node_weight[i] * ratio;

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        for (int j = x_lf; j < x_hf + 1; j++) {
            const float bin_x_l = j;
            const float bin_x_h = j + 1;
            float overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const float bin_y_l = k;
                const float bin_y_h = k + 1;
                float overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                atomicAdd(&aux_mat[j * num_bin_y + k], p_node_wght * overlap_area);
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

__global__ void density_map_cuda_normalize_node_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        // normalize bin_size_x and bin_size_y to 1
        normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
        normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
        normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
        normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
        normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
        if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
            normalize_node_info[i][3] - normalize_node_info[i][2] < 0 ||
            (node_size[i][0] < 1e-6 && node_size[i][1] < 1e-6)) {
            normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
        }
    }
} // END MODULE

//---------------------------------------------------------------------

__global__ void copyFromFloatAuxMat(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat_uint64[i] = static_cast<unsigned long long>(aux_mat[i] * scalar);
    }
}

__global__ void copyToFloatAuxMat(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat[i] = static_cast<float>(inv_scalar * aux_mat_uint64[i]);
    }
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    atomicAdd(&aux_mat[j * num_bin_y + k],
                              static_cast<unsigned long long>(weight * overlap_area * scalar));
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4) macro_density_map_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4] / 2;
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            const float x_c = (x_l + x_h) / 2;
            const float y_c = (y_l + y_h) / 2;
            const float ther = min((x_c - x_l), (y_c - y_l));
            const float x_l_ther = ther / 9 + x_l;
            const float x_h_ther = -ther / 9 + x_h;
            const float y_l_ther = ther / 9 + y_l;
            const float y_h_ther = -ther / 9 + y_h;
            float slope = weight / min((x_h_ther - x_c), (y_h_ther - y_c));
            int x_lf_ther = lround(floor(x_l_ther));
            int x_hf_ther = lround(floor(x_h_ther));
            int y_lf_ther = lround(floor(y_l_ther));
            int y_hf_ther = lround(floor(y_h_ther));
            int x_cf = lround(floor(x_c));
            int y_cf = lround(floor(y_c));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    float bin_weight;
                    if (j < x_cf && k < y_cf && (j - x_lf) > (k - y_lf))
                        bin_weight = (k - y_l_ther) * slope;
                    else if (j < x_cf && k < y_cf)
                        bin_weight = (j - x_l_ther) * slope;
                    else if (k < y_cf && (x_hf - j) > (k - y_lf))
                        bin_weight = (k - y_l_ther) * slope;
                    else if (k < y_cf)
                        bin_weight = (x_h_ther - j) * slope;
                    else if (j < x_cf && (j - x_lf) > (y_hf - k))
                        bin_weight = (y_h_ther - k) * slope;
                    else if (j < x_cf)
                        bin_weight = (j - x_l_ther) * slope;
                    else if ((x_hf - j) > (y_hf - k))
                        bin_weight = (y_h_ther - k) * slope;
                    else
                        bin_weight = (x_h_ther - j) * slope;
                    atomicAdd(&aux_mat[j * num_bin_y + k],
                              static_cast<unsigned long long>(bin_weight * overlap_area * scalar));
                }
            }
        }
    }
}


torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    if (deterministic) {
        // each bin size is pre-normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y;

        // use cache to save runtime
        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        static unsigned long long *aux_mat_uint64_ptr = nullptr;
        static int aux_mat_uint64_size = -1;
        if (aux_mat_uint64_ptr == nullptr) {
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        } else if (num_bin != aux_mat_uint64_size) {
            cudaFree(aux_mat_uint64_ptr);
            aux_mat_uint64_ptr = nullptr;
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        }
        //std::cout<<"?normalize_node_info info:"<<std::endl;
        //std::cout<<normalize_node_info<<std::endl;
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat_uint64_ptr,
            num_nodes,
            num_bin_x,
            num_bin_y,
            scalar);
        
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        //std::cout<<"aux_mat info:"<<std::endl;
        //std::cout<<aux_mat<<std::endl;
        // without cache, need a lot of cudaMallocAsync...
        // int cp_threads = 512;
        // int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        // unsigned long long *aux_mat_uint64 = nullptr;
        // cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        // copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
        //     normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        //     sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        //     aux_mat_uint64,
        //     num_nodes,
        //     num_bin_x,
        //     num_bin_y,
        //     scalar);
        // copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        density_map_cuda_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat.data_ptr<float>(),
            num_nodes,
            num_bin_x,
            num_bin_y);
    }

    return aux_mat;
}  // END MODULE

torch::Tensor macro_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    if (deterministic) {
        // each bin size is pre-normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y;

        // use cache to save runtime
        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        static unsigned long long *aux_mat_uint64_ptr = nullptr;
        static int aux_mat_uint64_size = -1;
        if (aux_mat_uint64_ptr == nullptr) {
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        } else if (num_bin != aux_mat_uint64_size) {
            cudaFree(aux_mat_uint64_ptr);
            aux_mat_uint64_ptr = nullptr;
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        }
        //std::cout<<"?normalize_node_info info:"<<std::endl;
        //std::cout<<normalize_node_info<<std::endl;
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        macro_density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat_uint64_ptr,
            num_nodes,
            num_bin_x,
            num_bin_y,
            scalar);
        
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        //std::cout<<"aux_mat info:"<<std::endl;
        //std::cout<<aux_mat<<std::endl;
        // without cache, need a lot of cudaMallocAsync...
        // int cp_threads = 512;
        // int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        // unsigned long long *aux_mat_uint64 = nullptr;
        // cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        // copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
        //     normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        //     sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        //     aux_mat_uint64,
        //     num_nodes,
        //     num_bin_x,
        //     num_bin_y,
        //     scalar);
        // copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        density_map_cuda_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat.data_ptr<float>(),
            num_nodes,
            num_bin_x,
            num_bin_y);
    }

    return aux_mat;
}  // END MODULE

//---------------------------------------------------------------------
__global__ void density_map_cuda_deterministic_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const float *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad_4part,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> macro_mask,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            float gradX = 0;
            float gradY = 0;
            for (int j = x_lf; j < x_hf + 1; j++) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf; k < y_hf + 1; k++) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    float grad_x = grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k] * overlap_area;
                    float grad_y = grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k] * overlap_area;
                    gradX += grad_x;
                    gradY += grad_y;
                    if(macro_mask[i])
                    {
                        int mid_x = (x_lf + x_hf + 1) >> 1;
                        int mid_y = (y_lf + y_hf + 1) >> 1;
                        if(j<=mid_x){
                            node_grad_4part[i][0]+=grad_x;
                        }else{
                            node_grad_4part[i][1]+=grad_x;
                        }
                        if(k<mid_y){
                            node_grad_4part[i][2]+=grad_y;
                        }else{
                            node_grad_4part[i][3]+=grad_y;
                        }
                    }
                }
            }
            node_grad[i][0] = grad_weight * weight * gradX;
            node_grad[i][1] = grad_weight * weight * gradY;
        }
    }
}

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        torch::Tensor& node_grad_4part,
                                        torch::Tensor& macro_mask,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        bool deterministic) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (deterministic) {
        int threads = 64;
        int blocks = (num_nodes + threads - 1) / threads;
        density_map_cuda_deterministic_backward_kernel<<<blocks, threads, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_mat.data_ptr<float>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_grad_4part.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            macro_mask.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            grad_weight,
            num_bin_x,
            num_bin_y,
            num_nodes);
    } else {
        int thread_count = 64;
        dim3 blockSize(2, 2, thread_count);
        int block_count = (num_nodes - 1 + thread_count) / thread_count;
        size_t shared_mem_size = sizeof(float) * thread_count * 2;
        density_map_cuda_backward_kernel<<<block_count, blockSize, shared_mem_size, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_mat.data_ptr<float>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_weight,
            num_bin_x,
            num_bin_y,
            num_nodes);
    }

    return node_grad;
}  // END MODULE

//---------------------------------------------------------------------
__global__ void copyFromFloatAuxMat2(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat_uint64[i] = static_cast<unsigned long long>(aux_mat[i] * scalar);
    }
}

__global__ void copyToFloatAuxMat2(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat[i] = static_cast<float>(inv_scalar * aux_mat_uint64[i]);
    }
}
__global__ void density_map_cuda_deterministic_forward_naive_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    unsigned long long *aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node,
    unsigned long long scalar) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float node_w = node_size[i][0];
        float node_h = node_size[i][1];
        float ratio = 1.0;
        if (clamp_node) {
            const float node_area = node_w * node_h;
            node_w = max(node_w, static_cast<float>(min_node_w));
            node_h = max(node_h, static_cast<float>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const float mgn = static_cast<float>(margin);
        const float num_bin_x_minus_mgn = static_cast<float>(num_bin_x) - mgn;
        const float num_bin_y_minus_mgn = static_cast<float>(num_bin_y) - mgn;
        const float small_mgn = static_cast<float>(margin * 0.1);
        float x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        float x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        float y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        float y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }
        const float p_node_wght = node_weight[i] * ratio;

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        for (int j = x_lf; j < x_hf + 1; j++) {
            const float bin_x_l = j;
            const float bin_x_h = j + 1;
            float overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const float bin_y_l = k;
                const float bin_y_h = k + 1;
                float overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                atomicAdd(&aux_mat[j * num_bin_y + k],
                          static_cast<unsigned long long>(p_node_wght * overlap_area * scalar));
            }
        }
    }
}
torch::Tensor density_map_cuda_forward_naive(torch::Tensor node_pos,
                                             torch::Tensor node_size,
                                             torch::Tensor node_weight,
                                             torch::Tensor unit_len,
                                             torch::Tensor aux_mat,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_nodes,
                                             float min_node_w,
                                             float min_node_h,
                                             float margin,
                                             bool clamp_node,
                                             bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    if (deterministic) {
        // each bin size is internally normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y;

        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        unsigned long long *aux_mat_uint64 = nullptr;
        cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        copyFromFloatAuxMat2<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        density_map_cuda_deterministic_forward_naive_kernel<<<blocks, threads, 0, stream>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            aux_mat_uint64,
            num_bin_x,
            num_bin_y,
            num_nodes,
            min_node_w,
            min_node_h,
            margin,
            clamp_node,
            scalar);
        copyToFloatAuxMat2<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        density_map_cuda_forward_naive_kernel<<<blocks, threads, 0, stream>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            aux_mat.data_ptr<float>(),
            num_bin_x,
            num_bin_y,
            num_nodes,
            min_node_w,
            min_node_h,
            margin,
            clamp_node);
    }

    return aux_mat;
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor density_map_cuda_normalize_node(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor unit_len,
                                              torch::Tensor normalize_node_info,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_nodes) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 128;
    const int blocks = (num_nodes + threads - 1) / threads;

    density_map_cuda_normalize_node_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        expand_ratio.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_bin_x,
        num_bin_y,
        num_nodes);

    return normalize_node_info;
}  // END MODULE

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> density_map,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_grad,
    int num_nodes,
    int num_bin_x,
    int num_bin_y) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][4];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            scalar_t total_wgt = 0;
            scalar_t total_area = (x_h - x_l) * (y_h - y_l);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);

                    scalar_t overlap_area = overlap_x * overlap_y;

                    gpuAtomicAdd(&node_grad[i], overlap_area * density_map[j][k] / total_area);
                }
            }
            // total_wgt /= total_area;
            // node_grad[i] = total_wgt;
        }
    }
}

torch::Tensor density_backward_cuda(torch::Tensor normalize_node_info,
                                    torch::Tensor sorted_node_map,
                                    torch::Tensor density_map,
                                    int num_bin_x,
                                    int num_bin_y,
                                    int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    auto node_grad =
        torch::zeros({num_nodes}, torch::dtype(normalize_node_info.dtype()).device(normalize_node_info.device()));

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_backward_cuda", ([&] {
                              density_backward_cuda_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  density_map.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_grad.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  num_nodes,
                                  num_bin_x,
                                  num_bin_y);
                          }));

    return node_grad;
}

template <typename scalar_t>
__global__ void applyFilter(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> density_map,
                            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aux_mat,
                            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> kernel,
                            int num_bin_x,
                            int num_bin_y,
                            int kernelWidth) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < num_bin_y && col < num_bin_x) {
        const int half = kernelWidth / 2;
        scalar_t blur = 0.0;
        for (int i = -half; i <= half; i++) {
            for (int j = -half; j <= half; j++) {
                const int y = max(0, min(num_bin_y - 1, row + i));
                const int x = max(0, min(num_bin_x - 1, col + j));

                scalar_t w = kernel[i + half][j + half];
                blur += w * density_map[x][y];
            }
        }
        gpuAtomicAdd(&aux_mat[col][row], blur);
    }
}

torch::Tensor apply_kernel_cuda(torch::Tensor density_map,
                                torch::Tensor aux_mat,
                                int num_bin_x,
                                int num_bin_y,
                                torch::Tensor kernel,
                                int kernelWidth) {
    cudaSetDevice(density_map.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int dim_block = 32;
    dim3 blockSize(dim_block, dim_block, 1);
    dim3 block_count(std::ceil(num_bin_x / dim_block), std::ceil(num_bin_y / dim_block), 1);

    AT_DISPATCH_ALL_TYPES(density_map.scalar_type(), "applyFilter", ([&] {
                              applyFilter<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  density_map.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  kernelWidth);
                          }));

    return aux_mat;
}