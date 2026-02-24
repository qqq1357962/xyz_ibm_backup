#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <ATen/cuda/Atomic.cuh>

#include "density_map_cuda.cuh"

namespace GP3D {

template <typename scalar_t>
__device__ scalar_t overlap(scalar_t x_l, scalar_t x_h, scalar_t bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
}

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

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_map_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area);
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void  density_map_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        // gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                            static_cast<unsigned long long>(weight * overlap_area * scalar));
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void  macro_density_map_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_macros,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_macros) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));
            const scalar_t x_c = (x_l + x_h) / 2;
            const scalar_t y_c = (y_l + y_h) / 2;
            const scalar_t ther = min((x_c - x_l), (y_c - y_l));
            const scalar_t x_l_ther = ther / 2 + x_l;
            const scalar_t x_h_ther = -ther / 2 + x_h;
            const scalar_t y_l_ther = ther / 2 + y_l;
            const scalar_t y_h_ther = -ther / 2 + y_h;
            // const scalar_t x_l_ther = x_l;
            // const scalar_t x_h_ther = x_h;
            // const scalar_t y_l_ther = y_l;
            // const scalar_t y_h_ther = y_h;
            float weight_scaler = 16;
            float slope = weight / min((x_h_ther - x_c), (y_h_ther - y_c)) / weight_scaler;
            int x_lf_ther = lround(floor(x_l_ther));
            int x_hf_ther = lround(floor(x_h_ther));
            int y_lf_ther = lround(floor(y_l_ther));
            int y_hf_ther = lround(floor(y_h_ther));
            int x_cf = lround(floor(x_c));
            int y_cf = lround(floor(y_c));

            // scalar_t area_scaler = (x_h - x_l) * (y_h - y_l) / ((x_h - x_l) * (y_h - y_l) + (x_h - x_l) * (y_h - y_l) / weight_scaler / 2 - min((x_c - x_l), (y_c - y_l)) * min((x_c - x_l), (y_c - y_l)) / weight_scaler / 4);

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t bin_weight;
                        // gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area
                        if (j < x_cf && k < y_cf && ((j - x_lf) > (k - y_lf)))
                            bin_weight = (k - y_l_ther) * slope;
                        else if (j < x_cf && k < y_cf)
                            bin_weight = (j - x_l_ther) * slope;
                        else if (k < y_cf && ((x_hf - j) > (k - y_lf)))
                            bin_weight = (k - y_l_ther) * slope;
                        else if (k < y_cf)
                            bin_weight = (x_h_ther - j) * slope;
                        else if (j < x_cf && ((j - x_lf) > (y_hf - k)))
                            bin_weight = (y_h_ther - k) * slope;
                        else if (j < x_cf)
                            bin_weight = (j - x_l_ther) * slope;
                        else if ((x_hf - j) > (y_hf - k))
                            bin_weight = (y_h_ther - k) * slope;
                        else
                            bin_weight = (x_h_ther - j) * slope;
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                            static_cast<unsigned long long>(bin_weight * overlap_area * scalar));
                        
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void  density_map_macro_overlay_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rotate_rate,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_macros,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_macros) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t macro_rotate_rate = rotate_rate[i];
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            scalar_t x_c = (x_l + x_h) / 2;
            scalar_t y_c = (y_l + y_h) / 2;
            scalar_t x_len = (x_h - x_l) / 2;
            scalar_t y_len = (y_h - y_l) / 2;
            scalar_t x_rotate_l = x_c - y_len * unit_len[1] / unit_len[0];
            scalar_t x_rotate_h = x_c + y_len * unit_len[1] / unit_len[0];
            scalar_t y_rotate_l = y_c - x_len * unit_len[0] / unit_len[1];
            scalar_t y_rotate_h = y_c + x_len * unit_len[0] / unit_len[1];
            int x_rotate_lf = lround(floor(x_rotate_l));
            int x_rotate_hf = lround(floor(x_rotate_h));
            int y_rotate_lf = lround(floor(y_rotate_l));
            int y_rotate_hf = lround(floor(y_rotate_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            x_rotate_lf = max(x_rotate_lf, 0);
            x_rotate_hf = min(x_rotate_hf, num_bin_x - 1);
            y_rotate_lf = max(y_rotate_lf, 0);
            y_rotate_hf = min(y_rotate_hf, num_bin_y - 1);


            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        // gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                            static_cast<unsigned long long>(weight * overlap_area * scalar * (1 - macro_rotate_rate)));
                    }
                }
            }

            for (int j = x_rotate_lf + threadIdx.y; j < x_rotate_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_rotate_l, x_rotate_h, bin_x_l);
                for (int k = y_rotate_lf + threadIdx.x; k < y_rotate_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_rotate_l, y_rotate_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        // gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                            static_cast<unsigned long long>(weight * overlap_area * scalar * macro_rotate_rate));
                    }
                }
            }
        }
    }
    else if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        // gpuAtomicAdd(&aux_mat[j][k][l], weight * overlap_area
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                            static_cast<unsigned long long>(weight * overlap_area * scalar));
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_map_macro_vertical_horizontal_overlap_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rotate_rate,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_macros) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_macros) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        const scalar_t macro_rotate_rate = rotate_rate[i];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            scalar_t x_c = (x_l + x_h) / 2;
            scalar_t y_c = (y_l + y_h) / 2;
            scalar_t x_len = (x_h - x_l) / 2;
            scalar_t y_len = (y_h - y_l) / 2;
            scalar_t x_rotate_l = x_c - y_len * unit_len[1] / unit_len[0];
            scalar_t x_rotate_h = x_c + y_len * unit_len[1] / unit_len[0];
            scalar_t y_rotate_l = y_c - x_len * unit_len[0] / unit_len[1];
            scalar_t y_rotate_h = y_c + x_len * unit_len[0] / unit_len[1];
            int x_rotate_lf = lround(floor(x_rotate_l));
            int x_rotate_hf = lround(floor(x_rotate_h));
            int y_rotate_lf = lround(floor(y_rotate_l));
            int y_rotate_hf = lround(floor(y_rotate_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            x_rotate_lf = max(x_rotate_lf, 0);
            x_rotate_hf = min(x_rotate_hf, num_bin_x - 1);
            y_rotate_lf = max(y_rotate_lf, 0);
            y_rotate_hf = min(y_rotate_hf, num_bin_y - 1);

            scalar_t macro_overlap = 0;
            scalar_t macro_overlap_90 = 0;

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                //for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp = grad_mat[j * num_bin_y * num_bin_z + k * num_bin_z + l];
                        
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        macro_overlap += tmp - overlap_area * (1 - macro_rotate_rate);
                    }
                }
            }

            for (int j = x_rotate_lf; j < x_rotate_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_rotate_l, x_rotate_h, bin_x_l);
                //for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                for (int k = y_rotate_lf; k < y_rotate_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_rotate_l, y_rotate_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp = grad_mat[j * num_bin_y * num_bin_z + k * num_bin_z + l];
                        
                        macro_overlap_90 += tmp - overlap_area * macro_rotate_rate;
                    }
                }
            }

            node_grad[i][0] = weight * macro_overlap;
            node_grad[i][1] = weight * macro_overlap_90;
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_map_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            extern __shared__ unsigned char grad_xy[];
            scalar_t *grad_x = (scalar_t *)grad_xy;
            scalar_t *grad_y = grad_x + blockDim.z;
            scalar_t *grad_z = grad_y + blockDim.z;  // FIXME:
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                grad_x[threadIdx.z] = grad_y[threadIdx.z] = grad_z[threadIdx.z] = 0;
            }
            __syncthreads();

            scalar_t part_grad_x = 0;
            scalar_t part_grad_y = 0;
            scalar_t part_grad_z = 0;

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_z = grad_mat[2 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        part_grad_x += overlap_area * tmp_x;
                        part_grad_y += overlap_area * tmp_y;
                        part_grad_z += overlap_area * tmp_z;
                    }
                }
            }
            gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            gpuAtomicAdd(&grad_z[threadIdx.z], part_grad_z);
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
                node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
                node_grad[i][2] = grad_weight * weight * grad_z[threadIdx.z];
            }
        }
    }
}

template <typename scalar_t>
__global__ void density_map_cuda_deterministic_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            scalar_t gradX = 0;
            scalar_t gradY = 0;
            scalar_t gradZ = 0;

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                //for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_z = grad_mat[2 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        gradX += overlap_area * tmp_x;
                        gradY += overlap_area * tmp_y;
                        gradZ += overlap_area * tmp_z;
                    }
                }
            }

            // gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            // gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            // gpuAtomicAdd(&grad_z[threadIdx.z], part_grad_z);
            // __syncthreads();
            node_grad[i][0] = grad_weight * weight * gradX;
            node_grad[i][1] = grad_weight * weight * gradY;
            node_grad[i][2] = grad_weight * weight * gradZ;
            // if (threadIdx.x == 0 && threadIdx.y == 0) {
                
            // }
        }
    }
}

template <typename scalar_t>
__global__ void density_map_overlay_cuda_deterministic_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rotate_state,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes,
    int num_macros) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_macros) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        const scalar_t macro_rotate_state = rotate_state[i];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            scalar_t x_c = (x_l + x_h) / 2;
            scalar_t y_c = (y_l + y_h) / 2;
            scalar_t x_len = (x_h - x_l) / 2;
            scalar_t y_len = (y_h - y_l) / 2;
            scalar_t x_rotate_l = x_c - y_len * unit_len[1] / unit_len[0];
            scalar_t x_rotate_h = x_c + y_len * unit_len[1] / unit_len[0];
            scalar_t y_rotate_l = y_c - x_len * unit_len[0] / unit_len[1];
            scalar_t y_rotate_h = y_c + x_len * unit_len[0] / unit_len[1];
            int x_rotate_lf = lround(floor(x_rotate_l));
            int x_rotate_hf = lround(floor(x_rotate_h));
            int y_rotate_lf = lround(floor(y_rotate_l));
            int y_rotate_hf = lround(floor(y_rotate_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            x_rotate_lf = max(x_rotate_lf, 0);
            x_rotate_hf = min(x_rotate_hf, num_bin_x - 1);
            y_rotate_lf = max(y_rotate_lf, 0);
            y_rotate_hf = min(y_rotate_hf, num_bin_y - 1);

            scalar_t gradX = 0;
            scalar_t gradY = 0;
            scalar_t gradZ = 0;

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                //for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * (1 - macro_rotate_state);
                        scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * (1 - macro_rotate_state);
                        scalar_t tmp_z = grad_mat[2 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * (1 - macro_rotate_state);
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        gradX += overlap_area * tmp_x;
                        gradY += overlap_area * tmp_y;
                        gradZ += overlap_area * tmp_z;
                    }
                }
            }

            for (int j = x_rotate_lf; j < x_rotate_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_rotate_l, x_rotate_h, bin_x_l);
                for (int k = y_rotate_lf; k < y_rotate_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_rotate_l, y_rotate_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * macro_rotate_state;
                        scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * macro_rotate_state;
                        scalar_t tmp_z = grad_mat[2 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l] * macro_rotate_state;
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        gradX += overlap_area * tmp_x;
                        gradY += overlap_area * tmp_y;
                        gradZ += overlap_area * tmp_z;
                    }
                }
            }

            // gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            // gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            // gpuAtomicAdd(&grad_z[threadIdx.z], part_grad_z);
            // __syncthreads();
            node_grad[i][0] = grad_weight * weight * gradX;
            node_grad[i][1] = grad_weight * weight * gradY;
            node_grad[i][2] = grad_weight * weight * gradZ;
            // if (threadIdx.x == 0 && threadIdx.y == 0) {
                
            // }
        }
    }
    else if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            scalar_t gradX = 0;
            scalar_t gradY = 0;
            scalar_t gradZ = 0;

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                //for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        scalar_t tmp_z = grad_mat[2 * num_bin_x * num_bin_y * num_bin_z + j * num_bin_y * num_bin_z +
                                                  k * num_bin_z + l];
                        // part_grad_x += overlap_area * grad_mat[0][j][k];
                        // part_grad_y += overlap_area * grad_mat[1][j][k];
                        gradX += overlap_area * tmp_x;
                        gradY += overlap_area * tmp_y;
                        gradZ += overlap_area * tmp_z;
                    }
                }
            }

            // gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            // gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            // gpuAtomicAdd(&grad_z[threadIdx.z], part_grad_z);
            // __syncthreads();
            node_grad[i][0] = grad_weight * weight * gradX;
            node_grad[i][1] = grad_weight * weight * gradY;
            node_grad[i][2] = grad_weight * weight * gradZ;
            // if (threadIdx.x == 0 && threadIdx.y == 0) {
                
            // }
        }
    }
}

//---------------------------------------------------------------------

template <typename scalar_t>
__global__ void density_map_cuda_forward_naive_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        if (node_weight[i] > 0) {
            scalar_t node_w = node_size[i][0];
            scalar_t node_h = node_size[i][1];
            scalar_t node_t = node_size[i][2];
            scalar_t ratio = 1.0;
            if (clamp_node) {
                const scalar_t node_area = node_w * node_h;
                node_w = max(node_w, static_cast<scalar_t>(min_node_w));
                node_h = max(node_h, static_cast<scalar_t>(min_node_h));
                ratio = node_area / (node_w * node_h);
            }
            const scalar_t mgn = static_cast<scalar_t>(margin);
            const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
            const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
            const scalar_t num_bin_z_minus_mgn = static_cast<scalar_t>(num_bin_z) - mgn;
            const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);

            scalar_t x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
            scalar_t x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
            scalar_t y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
            scalar_t y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
            scalar_t z_l = max((node_pos[i][2] - node_t / 2) / unit_len[2], mgn);
            scalar_t z_h = min((node_pos[i][2] + node_t / 2) / unit_len[2], num_bin_z_minus_mgn);
            x_l = min(x_l, num_bin_x_minus_mgn);
            x_h = max(x_h, mgn);
            y_l = min(y_l, num_bin_y_minus_mgn);
            y_h = max(y_h, mgn);
            z_l = min(z_l, num_bin_z_minus_mgn);
            z_h = max(z_h, mgn);
            if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
                return;
            }
            const scalar_t p_node_wght = node_weight[i] * ratio;
            
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        gpuAtomicAdd(&aux_mat[j][k][l], p_node_wght * overlap_area);
                    }
                }
            }
        }
    }
}  // END MODULE

template <typename scalar_t>
__global__ void density_map_cuda_deterministic_forward_naive_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    unsigned long long *aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node,
    unsigned long long scalar) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        if (node_weight[i] > 0) {
            scalar_t node_w = node_size[i][0];
            scalar_t node_h = node_size[i][1];
            scalar_t node_t = node_size[i][2];
            scalar_t ratio = 1.0;
            if (clamp_node) {
                const scalar_t node_area = node_w * node_h;
                node_w = max(node_w, static_cast<scalar_t>(min_node_w));
                node_h = max(node_h, static_cast<scalar_t>(min_node_h));
                ratio = node_area / (node_w * node_h);
            }
            const scalar_t mgn = static_cast<scalar_t>(margin);
            const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
            const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
            const scalar_t num_bin_z_minus_mgn = static_cast<scalar_t>(num_bin_z) - mgn;
            const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);

            scalar_t x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
            scalar_t x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
            scalar_t y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
            scalar_t y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
            scalar_t z_l = max((node_pos[i][2] - node_t / 2) / unit_len[2], mgn);
            scalar_t z_h = min((node_pos[i][2] + node_t / 2) / unit_len[2], num_bin_z_minus_mgn);
            x_l = min(x_l, num_bin_x_minus_mgn);
            x_h = max(x_h, mgn);
            y_l = min(y_l, num_bin_y_minus_mgn);
            y_h = max(y_h, mgn);
            z_l = min(z_l, num_bin_z_minus_mgn);
            z_h = max(z_h, mgn);
            if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
                return;
            }
            const scalar_t p_node_wght = node_weight[i] * ratio;
            
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            for (int j = x_lf; j < x_hf + 1; j++) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf; k < y_hf + 1; k++) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;
                        // gpuAtomicAdd(&aux_mat[j][k][l], p_node_wght * overlap_area);
                        atomicAdd(&aux_mat[j * num_bin_y * num_bin_z + k * num_bin_z +l],
                                static_cast<unsigned long long>(p_node_wght * overlap_area * scalar));
                    }
                }
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

template <typename scalar_t>
__global__ void density_map_cuda_normalize_node_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z,
    int num_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
        normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
        normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
        normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
        normalize_node_info[i][4] = (node_pos[i][2] - node_size[i][2] / 2) / unit_len[2];  // z_l
        normalize_node_info[i][5] = (node_pos[i][2] + node_size[i][2] / 2) / unit_len[2];  // z_h

        normalize_node_info[i][6] = node_weight[i] * expand_ratio[i];  // weight
        if (normalize_node_info[i][1] - normalize_node_info[i][0] <= 0 ||
            normalize_node_info[i][3] - normalize_node_info[i][2] <= 0 ||
            normalize_node_info[i][5] - normalize_node_info[i][4] <= 0) {
            normalize_node_info[i][6] = -normalize_node_info[i][6];  // we should ignore node whose weight <= 0
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_bin_z,
                                       int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    
    bool deterministic = true;
    if (deterministic) {
        // each bin size is pre-normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y * num_bin_z;

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
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
                              density_map_cuda_deterministic_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat_uint64_ptr,
                                  num_nodes,
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z,
                                  scalar);
                          }));
        
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
    } else {
        AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_forward", ([&] {
                              density_map_cuda_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  num_nodes,
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z);
                          }));
    }
    return aux_mat;
}  // END MODULE

//---------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> macro_overlay_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       torch::Tensor node_rotate_grad,
                                       torch::Tensor rotate_rate,
                                       torch::Tensor unit_len,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_bin_z,
                                       int num_nodes,
                                       int num_macros) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    
    bool deterministic = true;
    int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
    int scalar_bits = max(64 - max_value_bits, 0);
    unsigned long long scalar = (1UL << scalar_bits);
    float inv_scalar = 1.0 / static_cast<float>(scalar);
    int num_bin = num_bin_x * num_bin_y * num_bin_z;

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

    auto node_rotate_grad2 = torch::zeros_like(node_rotate_grad);

    copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
        density_map_macro_overlay_cuda_deterministic_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                rotate_rate.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                aux_mat_uint64_ptr,
                                num_nodes,
                                num_macros,
                                num_bin_x,
                                num_bin_y,
                                num_bin_z,
                                scalar);
                        }));

    copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if(t != 0) {
            printf("macro_overlay_density_map_cuda_forward flag1 error %d\n", t);
            assert(0);
            exit(0);
        }
    }
    TORCH_CHECK(normalize_node_info.is_cuda(), "normalize_node_info must be a CUDA tensor");
    TORCH_CHECK(rotate_rate.is_cuda(), "rotate_rate must be a CUDA tensor");
    TORCH_CHECK(unit_len.is_cuda(), "unit_len must be a CUDA tensor");
    TORCH_CHECK(aux_mat.is_cuda(), "aux_mat must be a CUDA tensor");
    TORCH_CHECK(sorted_node_map.is_cuda(), "sorted_node_map must be a CUDA tensor");
    TORCH_CHECK(node_rotate_grad2.is_cuda(), "node_rotate_grad2 must be a CUDA tensor");
    
    TORCH_CHECK(aux_mat.data_ptr() != nullptr, "aux_mat data_ptr is null");
    
    TORCH_CHECK(normalize_node_info.dim() == 2, "normalize_node_info must be 2D");
    TORCH_CHECK(rotate_rate.dim() == 1, "rotate_rate must be 1D");
    TORCH_CHECK(sorted_node_map.dim() == 1, "sorted_node_map must be 1D");
    
    TORCH_CHECK(normalize_node_info.size(0) >= num_macros, "normalize_node_info size mismatch");
    TORCH_CHECK(sorted_node_map.size(0) >= num_macros, "sorted_node_map size mismatch");
    TORCH_CHECK(normalize_node_info.is_contiguous(), "normalize_node_info must be contiguous");
    int64_t required_elements = (int64_t)num_bin_x * num_bin_y * num_bin_z;
    TORCH_CHECK(aux_mat.numel() >= required_elements, "aux_mat is smaller than bins volume");
    
    if (num_macros > 0) {
        auto map_min = sorted_node_map.min().item<int64_t>();
        auto map_max = sorted_node_map.max().item<int64_t>();
        TORCH_CHECK(map_max < normalize_node_info.size(0), "sorted_node_map max index out of range");
    }
    
    TORCH_CHECK(!normalize_node_info.isnan().any().item<bool>(), "normalize_node_info contains NaN");
    TORCH_CHECK(!normalize_node_info.isinf().any().item<bool>(), "normalize_node_info contains Inf");
    
    TORCH_CHECK(unit_len[0].item<double>() > 1e-9 && unit_len[1].item<double>() > 1e-9, "unit_len contains zero, causing inf coordinates");

    int threads = 64;
    int blocks = (num_macros + threads - 1) / threads;
    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
        density_map_macro_vertical_horizontal_overlap_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                rotate_rate.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                aux_mat.data_ptr<scalar_t>(),
                sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                node_rotate_grad2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                num_bin_x,
                num_bin_y,
                num_bin_z,
                num_macros);
    }));

    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("macro_overlay_density_map_cuda_forward failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
    }

    return {aux_mat, node_rotate_grad2};
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor macro_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_bin_z,
                                       int num_macros,
                                       int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    
    bool deterministic = true;
    if (deterministic) {
        // each bin size is pre-normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y * num_bin_z;

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
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
                              macro_density_map_cuda_deterministic_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat_uint64_ptr,
                                  num_nodes,
                                  num_macros,
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z,
                                  scalar);
                          }));
        
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
    }
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("macro_density_map_cuda_forward failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
    }
    return aux_mat;
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        torch::Tensor rotate_state,
                                        torch::Tensor unit_len,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_bin_z,
                                        int num_nodes,
                                        int num_macros) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    bool deterministic = true;
    if (deterministic) {
        int threads = 64;
        int blocks = (num_nodes + threads - 1) / threads;
        // AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
        //                       density_map_cuda_deterministic_backward_kernel<scalar_t>
        //                           <<<blocks, threads, 0, stream>>>(
        //                               normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        //                               grad_mat.data_ptr<scalar_t>(),
        //                               sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        //                               node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        //                               grad_weight,
        //                               num_bin_x,
        //                               num_bin_y,
        //                               num_bin_z,
        //                               num_nodes);
        //                   }));
        AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_deterministic_forward", ([&] {
                              density_map_overlay_cuda_deterministic_backward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_mat.data_ptr<scalar_t>(),
                                      sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      rotate_state.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                      unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                      grad_weight,
                                      num_bin_x,
                                      num_bin_y,
                                      num_bin_z,
                                      num_nodes,
                                      num_macros);
                          }));
    } else {
    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_backward", ([&] {
                              size_t shared_mem_size = sizeof(scalar_t) * thread_count * 2;
                              density_map_cuda_backward_kernel<scalar_t>
                                  <<<block_count, blockSize, shared_mem_size, stream>>>(
                                      normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_mat.data_ptr<scalar_t>(),
                                      sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_weight,
                                      num_bin_x,
                                      num_bin_y,
                                      num_bin_z,
                                      num_nodes);
                          }));

    }

    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("density_map_cuda_backward failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
    }

    return node_grad;
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor density_map_cuda_forward_naive(torch::Tensor node_pos,
                                             torch::Tensor node_size,
                                             torch::Tensor node_weight,
                                             torch::Tensor unit_len,
                                             torch::Tensor aux_mat,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_bin_z,
                                             int num_nodes,
                                             float min_node_w,
                                             float min_node_h,
                                             float margin,
                                             bool clamp_node) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;
    bool deterministic = true;
    if (deterministic) {
        // each bin size is internally normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y * num_bin_z;

        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        unsigned long long *aux_mat_uint64 = nullptr;
        cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_deterministic_forward_naive", ([&] {
                              density_map_cuda_deterministic_forward_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat_uint64,
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z,
                                  num_nodes,
                                  min_node_w,
                                  min_node_h,
                                  margin,
                                  clamp_node,
                                  scalar);
                          }));
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_forward_naive", ([&] {
                              density_map_cuda_forward_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z,
                                  num_nodes,
                                  min_node_w,
                                  min_node_h,
                                  margin,
                                  clamp_node);
                          }));
    }

    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("density_map_cuda_forward_naive failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
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
                                              int num_bin_z,
                                              int num_nodes) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 128;
    const int blocks = (num_nodes + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_normalize_node", ([&] {
                              density_map_cuda_normalize_node_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  expand_ratio.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_y,
                                  num_nodes);
                          }));
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("density_map_cuda_normalize_node failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
    }
    return normalize_node_info;
}  // END MODULE

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> density_map,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_grad,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    int num_bin_z) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][6];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            const scalar_t z_l = normalize_node_info[i][4];
            const scalar_t z_h = normalize_node_info[i][5];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            int z_lf = lround(floor(z_l));
            int z_hf = lround(floor(z_h));

            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            z_lf = max(z_lf, 0);
            z_hf = min(z_hf, num_bin_z - 1);

            scalar_t total_wgt = 0;
            scalar_t total_area = (x_h - x_l) * (y_h - y_l) * (z_h - z_l);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    for (int l = z_lf; l < z_hf + 1; l++) {  // TODO: idx z
                        scalar_t bin_z_l = static_cast<scalar_t>(l);
                        scalar_t overlap_z = overlap(z_l, z_h, bin_z_l);

                        scalar_t overlap_area = overlap_x * overlap_y * overlap_z;

                        gpuAtomicAdd(&node_grad[i], overlap_area * density_map[j][k][l] / total_area);
                    }
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
                                    int num_bin_z,
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
                                  density_map.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  node_grad.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  num_nodes,
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z);
                          }));
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("density_backward_cuda failed : %s\n", cudaGetErrorString(err));
            AT_ERROR("CUDA error: ", cudaGetErrorString(err));
            assert(0);
            exit(0);
        }
    }
    return node_grad;
}

template <typename scalar_t>
__global__ void applyFilter(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> density_map,
                            torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aux_mat,
                            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> kernel,
                            int num_bin_x,
                            int num_bin_y,
                            int num_bin_z,
                            int kernelWidth) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < num_bin_y && col < num_bin_x) {
        const int half = kernelWidth / 2;
        for (int k = 0; k < num_bin_z; k++) {
            scalar_t blur = 0.0;
            for (int i = -half; i <= half; i++) {
                for (int j = -half; j <= half; j++) {
                    const int y = max(0, min(num_bin_y - 1, row + i));
                    const int x = max(0, min(num_bin_x - 1, col + j));

                    scalar_t w = kernel[i + half][j + half];
                    blur += w * density_map[x][y][k];

                    // if (threadIdx.x == 0) {
                    //     printf("Index %d, %d, %d | %.2f, %.2f\n", i, j, k, w, density_map[x][y][k]);
                    // }
                }
            }
            gpuAtomicAdd(&aux_mat[col][row][k], blur);
            // gpuAtomicAdd(&aux_mat[col][row][k], density_map[col][row][k]);
        }
    }
}

torch::Tensor apply_kernel_cuda(torch::Tensor density_map,
                                torch::Tensor aux_mat,
                                int num_bin_x,
                                int num_bin_y,
                                int num_bin_z,
                                torch::Tensor kernel,
                                int kernelWidth) {
    cudaSetDevice(density_map.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int dim_block = std::min(32, num_bin_x);
    dim3 blockSize(dim_block, dim_block, 1);
    dim3 block_count(std::ceil(num_bin_x / dim_block), std::ceil(num_bin_y / dim_block), 1);

    AT_DISPATCH_ALL_TYPES(density_map.scalar_type(), "applyFilter", ([&] {
                              applyFilter<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  density_map.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  kernel.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_bin_z,
                                  kernelWidth);
                          }));

    return aux_mat;
}

}  // namespace GP3D