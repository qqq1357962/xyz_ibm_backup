#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <ATen/cuda/Atomic.cuh>

#include "density_smooth_cuda.cuh"

template <typename scalar_t>
__device__ scalar_t overlap(scalar_t x_l, scalar_t x_h, scalar_t bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t bell_fn(scalar_t d, scalar_t a, scalar_t b, scalar_t w) {
    if (d <= w / 2 + 1)
        return 1 - a * d * d;
    else if (d >= w / 2 + 1 && d <= w / 2 + 2)
        return b * (d - w / 2 - 2) * (d - w / 2 - 2);
    else
        return 0;
}

//---------------------------------------------------------------------

template <typename scalar_t>
__global__ void density_smooth_cuda_normalize_node_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        const scalar_t node_x = node_pos[i][0] / unit_len[0];
        const scalar_t node_y = node_pos[i][1] / unit_len[1];
        const scalar_t node_w = node_size[i][0] / unit_len[0];
        const scalar_t node_h = node_size[i][1] / unit_len[1];
        const scalar_t p_node_wght = node_weight[i];

        const scalar_t ax = (scalar_t)4 / ((node_w + 2) * (node_w + 4));
        const scalar_t bx = (scalar_t)2 / (node_w * (node_w + 4));

        const scalar_t ay = (scalar_t)4 / ((node_h + 2) * (node_h + 4));
        const scalar_t by = (scalar_t)2 / (node_h * (node_h + 4));

        const scalar_t x_l = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0] - 2; // FIXME: window
        const scalar_t x_h = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0] + 2;
        const scalar_t y_l = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1] - 2;
        const scalar_t y_h = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1] + 2;

        // const scalar_t x_l = node_x - node_w / 2;
        // const scalar_t x_h = node_x + node_w / 2;
        // const scalar_t y_l = node_y - node_h / 2;
        // const scalar_t y_h = node_y + node_h / 2;

        normalize_node_info[i][0] = x_l;  // x_l
        normalize_node_info[i][1] = x_h;  // x_h
        normalize_node_info[i][2] = y_l;  // y_l
        normalize_node_info[i][3] = y_h;  // y_h
        normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
        if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
            normalize_node_info[i][3] - normalize_node_info[i][2] < 0) {
            normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
        }

        // normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
        // normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
        // normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
        // normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
        // normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
        // if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
        //     normalize_node_info[i][3] - normalize_node_info[i][2] < 0) {
        //     normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
        // }

        normalize_node_info[i][5] = node_x;  // node_x
        normalize_node_info[i][6] = node_y;  // node_y
        normalize_node_info[i][7] = node_w;  // node_w
        normalize_node_info[i][8] = node_h;  // node_h

        normalize_node_info[i][9] = ax;   // ax
        normalize_node_info[i][10] = bx;  // bx
        normalize_node_info[i][11] = ay;  // ay
        normalize_node_info[i][12] = by;  // by
    }
    // if (i < num_nodes) {
    //     normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
    //     normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
    //     normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
    //     normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
    //     normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
    //     if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
    //         normalize_node_info[i][3] - normalize_node_info[i][2] < 0) {
    //         normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
    //     }
    // }
}  // END MODULE


torch::Tensor density_smooth_cuda_normalize_node(torch::Tensor node_pos,
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

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_smooth_cuda_normalize_node", ([&] {
                              density_smooth_cuda_normalize_node_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  expand_ratio.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_nodes);
                          }));

    return normalize_node_info;
}  // END MODULE

//---------------------------------------------------------------------

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_smooth_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][4];
        if (weight > 0) {
            // const scalar_t x_l = normalize_node_info[i][0];
            // const scalar_t x_h = normalize_node_info[i][1];
            // const scalar_t y_l = normalize_node_info[i][2];
            // const scalar_t y_h = normalize_node_info[i][3];
            // int x_lf = lround(floor(x_l));
            // int x_hf = lround(floor(x_h));
            // int y_lf = lround(floor(y_l));
            // int y_hf = lround(floor(y_h));
            // x_lf = max(x_lf, 0);
            // x_hf = min(x_hf, num_bin_x - 1);
            // y_lf = max(y_lf, 0);
            // y_hf = min(y_hf, num_bin_y - 1);

            // for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
            //     scalar_t bin_x_l = static_cast<scalar_t>(j);
            //     scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
            //     for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
            //         scalar_t bin_y_l = static_cast<scalar_t>(k);
            //         scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
            //         scalar_t overlap_area = overlap_x * overlap_y;
            //         gpuAtomicAdd(&aux_mat[j][k], weight * overlap_area);
            //     }
            // }
            const scalar_t p_node_wght = weight;
            const scalar_t node_x = normalize_node_info[i][5];
            const scalar_t node_y = normalize_node_info[i][6];
            const scalar_t node_w = normalize_node_info[i][7];
            const scalar_t node_h = normalize_node_info[i][8];

            const scalar_t ax = normalize_node_info[i][9];
            const scalar_t bx = normalize_node_info[i][10];
            const scalar_t ay = normalize_node_info[i][11];
            const scalar_t by = normalize_node_info[i][12];

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

            // if (threadIdx.x == 0) {
            //     printf("node %d ---- weight %.2f\n", i, weight);
            //     printf("%d ---- %d-%d, %d-%d\n", i, x_lf, x_hf, y_lf, y_hf);
            // }

            scalar_t psum = 0;
            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                const scalar_t dx = abs(j + 0.5 - node_x);
                const scalar_t px = bell_fn(dx, ax, bx, node_w);
                // if (threadIdx.x == 0) {
                //     printf("%.2f | %.2f | %.2f\n", ax, bx, node_w);
                //     printf("%.2f -> %.2f\n", dx, px);
                // }
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    const scalar_t dy = abs(k + 0.5 - node_y);
                    const scalar_t py = bell_fn(dy, ay, by, node_h);
                    scalar_t overlap_area = px * py;
                    psum = psum + overlap_area;
                }
            }


            scalar_t pcoef = node_w * node_h / psum / (scalar_t)4; // FIXME:
            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                // scalar_t bin_x_l = static_cast<scalar_t>(j);
                // scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);

                const scalar_t dx = abs(j + 0.5 - node_x);
                const scalar_t px = bell_fn(dx, ax, bx, node_w);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    // scalar_t bin_y_l = static_cast<scalar_t>(k);
                    // scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);

                    const scalar_t dy = abs(k + 0.5 - node_y);
                    const scalar_t py = bell_fn(dy, ay, by, node_h);
                    scalar_t overlap_area = px * py;
                    gpuAtomicAdd(&aux_mat[j][k], p_node_wght * overlap_area * pcoef);

                    // overlap_area = overlap_x * overlap_y;
                    // gpuAtomicAdd(&aux_mat[j][k], p_node_wght * overlap_area);
                }
            }
        }
    }
}

// template <typename scalar_t>
// __global__ void sum_density_map_kernel(
//     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_density_map,
//     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> temp_mat,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aux_mat,
//     int num_bin_x,
//     int num_bin_y) {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     const int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < num_bin_x && j < num_bin_y) {
//         gpuAtomicAdd(&aux_mat[i][j], init_density_map[i][j] + temp_mat[i][j]);
//     }
// }

torch::Tensor density_smooth_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes) {
    // cudaSetDevice(normalize_node_info.get_device());
    // auto stream = at::cuda::getCurrentCUDAStream();

    // // auto aux_mat = torch::zeros({num_bin_x, num_bin_y}, torch::dtype(node_pos.dtype()).device(node_pos.device()));
    // // auto temp_mat = torch::zeros({num_bin_x, num_bin_y}, torch::dtype(node_pos.dtype()).device(node_pos.device()));

    // // const int threads = 128;
    // // const int blocks = (num_nodes + threads - 1) / threads;

    // // AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_smooth_cuda_forward", ([&] {
    // //                           density_smooth_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
    // //                               node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // //                               node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // //                               node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    // //                               unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    // //                               aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // //                               num_bin_x,
    // //                               num_bin_y,
    // //                               num_nodes);
    // //                       }));
    
    // // // dim3 threadsPerBlock(8, 8);
    // // // dim3 numBlocks(num_bin_x / threadsPerBlock.x, num_bin_y / threadsPerBlock.y);
    // // // AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "sum_density_map", ([&] {
    // // //                           sum_density_map_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
    // // //                               init_density_map.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // // //                               temp_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // // //                               aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    // // //                               num_bin_x,
    // // //                               num_bin_y);
    // // //                       }));

    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_smooth_cuda_forward", ([&] {
                              density_smooth_cuda_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_nodes);
                          }));

    return aux_mat;
}  // END MODULE

//---------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ scalar_t bell_fn_grad(scalar_t d, scalar_t a, scalar_t b, scalar_t w) {
    if (d <= w / 2 + 1)
        // return 1 - a * d * d;
        return -2 * a * d;
    else if (d >= w / 2 + 1 && d <= w / 2 + 2)
        // return b * (d - w / 2 - 2) * (d - w / 2 - 2);
        return 2 * b * (d - w / 2 - 2);
    else
        return 0;
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_smooth_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][4];
        if (weight > 0) {
            const scalar_t p_node_wght = weight;
            const scalar_t node_x = normalize_node_info[i][5];
            const scalar_t node_y = normalize_node_info[i][6];
            const scalar_t node_w = normalize_node_info[i][7];
            const scalar_t node_h = normalize_node_info[i][8];

            const scalar_t ax = normalize_node_info[i][9];
            const scalar_t bx = normalize_node_info[i][10];
            const scalar_t ay = normalize_node_info[i][11];
            const scalar_t by = normalize_node_info[i][12];

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

            extern __shared__ unsigned char grad_xy[];
            scalar_t *grad_x = (scalar_t *)grad_xy;
            scalar_t *grad_y = grad_x + blockDim.z;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                grad_x[threadIdx.z] = grad_y[threadIdx.z] = 0;
            }
            __syncthreads();

            scalar_t part_grad_x = 0;
            scalar_t part_grad_y = 0;


            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                const scalar_t dx = -(j + 0.5 - node_x);
                const scalar_t px = bell_fn(dx, ax, bx, node_w);
                const scalar_t dpx = bell_fn_grad(dx, ax, bx, node_w);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    scalar_t overlap_area = overlap_x * overlap_y;
                    
                    const scalar_t dy = -(k + 0.5 - node_y);
                    const scalar_t py = bell_fn(dy, ay, by, node_h);
                    const scalar_t dpy = bell_fn_grad(dy, ay, by, node_h);
                    
                    scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k];

                    part_grad_x += overlap_area * tmp_x;
                    part_grad_y += overlap_area * tmp_y;
                }
            }

            gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
                node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
            }

        }
        
    }
}

torch::Tensor density_smooth_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // const int threads = 256;
    // const int blocks = (num_nodes + threads - 1) / threads;

    // AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_smooth_cuda_backward", ([&] {
    //                           density_smooth_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
    //                               node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               grad_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    //                               node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               num_bin_x,
    //                               num_bin_y,
    //                               num_nodes);
    //                       }));

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_smooth_cuda_backward", ([&] {
                              size_t shared_mem_size = sizeof(scalar_t) * thread_count * 2;
                              density_smooth_cuda_backward_kernel<scalar_t>
                                  <<<block_count, blockSize, shared_mem_size, stream>>>(
                                      normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_mat.data_ptr<scalar_t>(),
                                      sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_weight,
                                      num_bin_x,
                                      num_bin_y,
                                      num_nodes);
                          }));

    return node_grad;
}