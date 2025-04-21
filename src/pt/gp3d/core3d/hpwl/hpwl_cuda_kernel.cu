#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cmath>

#include "hpwl_cuda.cuh"

namespace GP3D {

template <typename scalar_t>
__global__ void node_pos_to_pin_pos_cross_chip_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> node_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // pin index
    const int i = index;  // pin index
    if (i < num_pins) {
        for (int c = 0; c < 3; c++) {
            int64_t node_id = pin_id2node_id[i];
            pin_pos[i][c] += node_pos[node_id][c];
            pin_die[i] = node_die[node_id];
        }
        // const int c = index & 1;  // channel index
    }
}

template <typename scalar_t>
__global__ void node_pos_to_pin_pos_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // pin index
    const int i = index;  // pin index
    if (i < num_pins) {
        for (int c = 0; c < 3; c++) {
            int64_t node_id = pin_id2node_id[i];
            pin_pos[i][c] += node_pos[node_id][c];
        }
        // const int c = index & 1;  // channel index
    }
}

torch::Tensor node_pos_to_pin_pos_cuda_forward(torch::Tensor node_pos,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    // pin_pos == node_pos + pin_rel_cpos

    const auto num_pins = pin_id2node_id.size(0);

    auto pin_pos = pin_rel_cpos.clone();  // pin

    const int threads = 64;
    const int blocks = (num_pins + threads - 1) / threads;  // TODO:

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_pins);
                          }));

    return pin_pos;
}

template <typename scalar_t>
__global__ void masked_scale_hpwl_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hpwl_scale,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            for (int c = 0; c < 3; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = roundf(abs(x_max - x_min) * hpwl_scale[c]);
            }
        }
    }
}

// Following TCAD-DreamPLACE

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_linear_model_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            for (int c = 0; c < 2; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                scalar_t xexp_x_sum = 0;
                scalar_t xexp_nx_sum = 0;
                scalar_t exp_x_sum = 0;
                scalar_t exp_nx_sum = 0;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t exp_x = exp((xx - x_max) * inv_gamma);
                    scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;
                }

                scalar_t wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
                partial_wa_wl[i][c] = wl;

                scalar_t b_x = inv_gamma / (exp_x_sum);
                scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
                scalar_t b_nx = -inv_gamma / (exp_nx_sum);
                scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t exp_x = exp((xx - x_max) * inv_gamma);
                    scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

                    pin_grad[hyperedge_list[idx]][c] =
                        (c == 2 ? w : 1) * ((a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx);
                }
            }
            // use x, y information to influence z direction
            for (int c = 0; c < 2; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t m = 0;
                scalar_t b = 0;
                scalar_t x_sum = 0;
                scalar_t z_sum = 0;
                scalar_t xz_sum = 0;
                scalar_t x_squre_sum = 0;
                int n = 0;
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if (pin_die[hyperedge_list[idx]] == -1) continue;
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_sum += xx;
                    z_sum += pin_pos[hyperedge_list[idx]][2];
                    xz_sum += xx * pin_pos[hyperedge_list[idx]][2];
                    x_squre_sum += xx*xx;
                    n++;
                }
                scalar_t x_mean = x_sum / n;
                scalar_t z_mean = z_sum / n;
                m = (xz_sum - x_mean * z_mean * n)/(x_squre_sum - n * x_mean * x_mean);
                b = z_mean - m * x_mean;
                // printf("m= %f, b= %f\n", m, b);
                // partial_hpwl[i][c] = abs(x_max - x_min);// when will partial_hpwl[i][2] multiply by 0?
                
                partial_hpwl[i][c] = 0;
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t zz = pin_pos[hyperedge_list[idx]][2];
                    scalar_t distance = abs(m*xx+b-zz);
                    // printf("m*xx+b = %f, zz = %f\n", m*xx+b, zz);
                    if(c==0)
                    {
                        //(m*xx+b-zz)>=0 -> need to move upward, grad<0
                        // pin_grad[hyperedge_list[idx]][2] = -0.01;
                        if(m*xx+b-zz!=0)
                        {
                            pin_grad[hyperedge_list[idx]][2] = -((m*xx+b-zz)>=0) * 0.002;
                        } 
                    }else{
                        // pin_grad[hyperedge_list[idx]][2] += -0.01;
                        if(m*xx+b-zz!=0)
                        {
                            pin_grad[hyperedge_list[idx]][2] += -((m*xx+b-zz)>=0) * 0.002;
                        }
                    }
                    if(c==0&&idx==start_idx)
                    {
                        partial_hpwl[i][c] = distance;
                    }else{
                        partial_hpwl[i][c] += distance;
                    }
                }
            }
        }
    }
}

// __device__ float project_x(float x, float z, float z_range, float via_x, int project_layer){
//     float ratio = z / z_range;
//     if(project_layer==0) ratio = 1-ratio;
//     return via_x + ratio * (x - via_x);
// }



// template <typename scalar_t>
// __global__ void wa_wirelength_hpwl_project_model_cuda_kernel(
//     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
//     const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
//     const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
//     const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
//     const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
//     const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
//     int num_nets,
//     float inv_gamma,
//     float z_range) {
//     const int index = blockIdx.x * blockDim.x + threadIdx.x;
//     // const int i = index >> 1;  // net index
//     const int i = index;  // pin index
//     if (i < num_nets && net_mask[i]) {
//         // const int c = index & 1;  // channel index
//         scalar_t w = net_weight[i];
//         int64_t start_idx = 0;
//         if (i != 0) {
//             start_idx = hyperedge_list_end[i - 1];
//         }
//         int64_t end_idx = hyperedge_list_end[i];
//         if (end_idx != start_idx) {
//             for (int die_id = 0; die_id < 2; die_id++) {
//             for (int c = 0; c < 2; c++) {
//                 int64_t pin_id = hyperedge_list[start_idx];
//                 scalar_t x_min = pin_pos[pin_id][c];
//                 scalar_t x_max = pin_pos[pin_id][c];
//                 for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     x_min = min(xx, x_min);
//                     x_max = max(xx, x_max);
//                 }
//                 partial_hpwl[i][c] = abs(x_max - x_min);

//                 scalar_t xexp_x_sum = 0;
//                 scalar_t xexp_nx_sum = 0;
//                 scalar_t exp_x_sum = 0;
//                 scalar_t exp_nx_sum = 0;

//                 for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                     // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
//                     if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     scalar_t exp_x = exp((xx - x_max) * inv_gamma);
//                     scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

//                     xexp_x_sum += xx * exp_x;
//                     xexp_nx_sum += xx * exp_nx;
//                     exp_x_sum += exp_x;
//                     exp_nx_sum += exp_nx;
//                 }

//                 scalar_t wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
//                 partial_wa_wl[i][c] = wl;

//                 scalar_t b_x = inv_gamma / (exp_x_sum);
//                 scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
//                 scalar_t b_nx = -inv_gamma / (exp_nx_sum);
//                 scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

//                 for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                     if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
//                     // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     scalar_t exp_x = exp((xx - x_max) * inv_gamma);
//                     scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

//                     pin_grad[hyperedge_list[idx]][c] =
//                         (c == 2 ? w : 1) * ((a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx);
//                 }
//             }
//             // use x, y information to influence z direction
//             for (int c = 0; c < 2; c++) {
//                 int64_t pin_id = hyperedge_list[start_idx];
//                 scalar_t m = 0;
//                 scalar_t b = 0;
//                 scalar_t x_sum = 0;
//                 scalar_t z_sum = 0;
//                 scalar_t xz_sum = 0;
//                 scalar_t x_squre_sum = 0;
//                 int n = 0;
//                 for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                     if (pin_die[hyperedge_list[idx]] == -1) continue;
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     x_sum += xx;
//                     z_sum += pin_pos[hyperedge_list[idx]][2];
//                     xz_sum += xx * pin_pos[hyperedge_list[idx]][2];
//                     x_squre_sum += xx*xx;
//                     n++;
//                 }
//                 scalar_t x_mean = x_sum / n;
//                 scalar_t z_mean = z_sum / n;
//                 m = (xz_sum - x_mean * z_mean * n)/(x_squre_sum - n * x_mean * x_mean);
//                 b = z_mean - m * x_mean;
//                 // printf("m= %f, b= %f\n", m, b);
//                 // partial_hpwl[i][c] = abs(x_max - x_min);// when will partial_hpwl[i][2] multiply by 0?
                
//                 partial_hpwl[i][c] = 0;
//                 for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                     // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
//                     if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     scalar_t zz = pin_pos[hyperedge_list[idx]][2];
//                     scalar_t distance = abs(m*xx+b-zz);
//                     // printf("m*xx+b = %f, zz = %f\n", m*xx+b, zz);
//                     if(c==0)
//                     {
//                         //(m*xx+b-zz)>=0 -> need to move upward, grad<0
//                         // pin_grad[hyperedge_list[idx]][2] = -0.01;
//                         if(m*xx+b-zz!=0)
//                         {
//                             pin_grad[hyperedge_list[idx]][2] = -((m*xx+b-zz)>=0) * 0.002;
//                         } 
//                     }else{
//                         // pin_grad[hyperedge_list[idx]][2] += -0.01;
//                         if(m*xx+b-zz!=0)
//                         {
//                             pin_grad[hyperedge_list[idx]][2] += -((m*xx+b-zz)>=0) * 0.002;
//                         }
//                     }
//                     if(c==0&&idx==start_idx)
//                     {
//                         partial_hpwl[i][c] = distance;
//                     }else{
//                         partial_hpwl[i][c] += distance;
//                     }
//                 }
//             }
//         }
//     }
// }

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> macro_mask,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ratio_multiply,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            float macro = 0;
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                if (macro_mask[pin_id2node_id[hyperedge_list[idx]]] == 1)
                {
                    macro = 1;
                    break;
                }
            }
            for (int c = 0; c < 3; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                scalar_t slack = abs(x_max - x_min);
                double_t xexp_x_sum = 0;
                double_t xexp_nx_sum = 0;
                double_t exp_x_sum = 0;
                double_t exp_nx_sum = 0;



                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    // printf("slack: %f\n", slack);
                    // printf("inv_gamma: %f\n", inv_gamma);
                    // printf("XMAX: %f, %f\n", x_max, xx);
                    // printf("XMIN: %f, %f\n", x_min, xx);
                    // printf("XX: %f, %f\n", exp_x, xx);
                    // printf("XX_N: %f, %f\n", exp_nx, xx);

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;
                }

                if (exp_x_sum < 1) {
                    printf("bug_exp_x_sum: %f\n", exp_x_sum);
                }

                if (exp_nx_sum < 1) {
                    printf("bug_exp_nx_sum: %f\n", exp_nx_sum);
                }

                scalar_t s_x = xexp_x_sum / exp_x_sum;
                scalar_t ns_x = xexp_nx_sum / exp_nx_sum;

                scalar_t wl = s_x - ns_x;
                partial_wa_wl[i][c] = wl;
                scalar_t scale = 1;


                scalar_t b_x = scale * inv_gamma / (exp_x_sum);
                scalar_t a_x = scale * (1.0 - inv_gamma * s_x) / exp_x_sum;
                scalar_t b_nx = scale * (-inv_gamma) / (exp_nx_sum);
                scalar_t a_nx = scale * (1.0 + inv_gamma * ns_x) / exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    scalar_t grad = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
                    // if ((grad >= 0 && grad < 0.0001) || grad > 10000 || (grad <= 0 && grad > -0.0001) || grad < -10000) {
                    //     printf("grad: %f\n", grad);
                    // }

                    if (c != 2) {
                        pin_grad[hyperedge_list[idx]][c] = grad;
                    } else {
                        pin_grad[hyperedge_list[idx]][c] += w * grad;
                        
                        if (macro_mask[pin_id2node_id[hyperedge_list[idx]]] == 1)
                            pin_grad[hyperedge_list[idx]][c] -= ratio_multiply[pin_id2node_id[hyperedge_list[idx]]];
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_cuda_kernel_with_pin_slide(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_rel_cpos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> pin_slide,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> macro_mask,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ratio_multiply,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            float macro = 0;
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                if (macro_mask[pin_id2node_id[hyperedge_list[idx]]] == 1)
                {
                    macro = 1;
                    break;
                }
            }
            for (int c = 0; c < 3; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                scalar_t slack = abs(x_max - x_min);
                double_t xexp_x_sum = 0;
                double_t xexp_nx_sum = 0;
                double_t exp_x_sum = 0;
                double_t exp_nx_sum = 0;



                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;
                }

                scalar_t s_x = xexp_x_sum / exp_x_sum;
                scalar_t ns_x = xexp_nx_sum / exp_nx_sum;

                scalar_t wl = s_x - ns_x;
                partial_wa_wl[i][c] = wl;
                scalar_t scale = 1;


                scalar_t b_x = scale * inv_gamma / (exp_x_sum);
                scalar_t a_x = scale * (1.0 - inv_gamma * s_x) / exp_x_sum;
                scalar_t b_nx = scale * (-inv_gamma) / (exp_nx_sum);
                scalar_t a_nx = scale * (1.0 + inv_gamma * ns_x) / exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    double_t exp_x;
                    double_t exp_nx;

                    scalar_t long_side_pin_rel_cpos = 1 / sqrt(pin_rel_cpos[pin_id][0] * pin_rel_cpos[pin_id][0] + pin_rel_cpos[pin_id][1] * pin_rel_cpos[pin_id][1]);

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    scalar_t grad = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

                    if (c != 2) {
                        pin_grad[pin_id][c] = grad;
                        if (macro_mask[pin_id2node_id[pin_id]] == 1) {
                            pin_slide[pin_id] += (-1) * grad * long_side_pin_rel_cpos * pin_rel_cpos[pin_id][c];
                        }
                    } else {
                        pin_grad[pin_id][c] += w * grad;
                        
                        if (macro_mask[pin_id2node_id[pin_id]] == 1)
                            pin_grad[pin_id][c] -= ratio_multiply[pin_id2node_id[pin_id]];
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_cuda_kernel_with_pin_slide_and_pin_orient(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_rel_cpos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> pin_slide,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> pin_orient,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> macro_mask,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ratio_multiply,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            float macro = 0;
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                if (macro_mask[pin_id2node_id[hyperedge_list[idx]]] == 1)
                {
                    macro = 1;
                    break;
                }
            }
            for (int c = 0; c < 3; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                scalar_t slack = abs(x_max - x_min);
                double_t xexp_x_sum = 0;
                double_t xexp_nx_sum = 0;
                double_t exp_x_sum = 0;
                double_t exp_nx_sum = 0;



                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;
                }

                scalar_t s_x = xexp_x_sum / exp_x_sum;
                scalar_t ns_x = xexp_nx_sum / exp_nx_sum;

                scalar_t wl = s_x - ns_x;
                partial_wa_wl[i][c] = wl;
                scalar_t scale = 1;


                scalar_t b_x = scale * inv_gamma / (exp_x_sum);
                scalar_t a_x = scale * (1.0 - inv_gamma * s_x) / exp_x_sum;
                scalar_t b_nx = scale * (-inv_gamma) / (exp_nx_sum);
                scalar_t a_nx = scale * (1.0 + inv_gamma * ns_x) / exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    double_t exp_x;
                    double_t exp_nx;

                    scalar_t long_side_pin_rel_cpos = 1 / sqrt(pin_rel_cpos[pin_id][0] * pin_rel_cpos[pin_id][0] + pin_rel_cpos[pin_id][1] * pin_rel_cpos[pin_id][1]);

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    scalar_t grad = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

                    if (c != 2) {
                        pin_grad[pin_id][c] = grad;
                        if (macro_mask[pin_id2node_id[pin_id]] == 1) {
                            pin_slide[pin_id] += (-1) * grad * long_side_pin_rel_cpos * pin_rel_cpos[pin_id][c];
                            pin_orient[pin_id] += ((c % 2 == 0) ? -1 : 1) * grad * long_side_pin_rel_cpos * pin_rel_cpos[pin_id][1 - c];
                        }
                    } else {
                        pin_grad[pin_id][c] += w * grad;
                        
                        if (macro_mask[pin_id2node_id[pin_id]] == 1)
                            pin_grad[pin_id][c] -= ratio_multiply[pin_id2node_id[pin_id]];
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_multi_layer_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> macro_mask,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            float macro = 0;
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                if (macro_mask[pin_id2node_id[idx]] == 1)
                {
                    macro = 1;
                    break;
                }
            }
            for (int c = 0; c < 3; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                scalar_t top_x_min = 0;
                scalar_t top_x_max = 0;
                scalar_t bot_x_min = 0;
                scalar_t bot_x_max = 0;
                int top_nonempty = 0;
                int bot_nonempty = 0;
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                    if ( pin_die[hyperedge_list[idx]] == 1 ) {
                        if (top_x_min == 0 || top_x_min > xx) { top_x_min = xx; }
                        if (top_x_max < xx) { top_x_max = xx; }
                        top_nonempty = 1;
                    }
                    if ( pin_die[hyperedge_list[idx]] == 0 ) {
                        if (bot_x_min == 0 || bot_x_min > xx) { bot_x_min = xx; }
                        if (bot_x_max < xx) { bot_x_max = xx; }
                        bot_nonempty = 1;
                    }
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                double_t xexp_x_sum = 0;
                double_t xexp_nx_sum = 0;
                double_t exp_x_sum = 0;
                double_t exp_nx_sum = 0;
                double_t top_xexp_x_sum = 0;
                double_t top_xexp_nx_sum = 0;
                double_t top_exp_x_sum = 0;
                double_t top_exp_nx_sum = 0;
                double_t bot_xexp_x_sum = 0;
                double_t bot_xexp_nx_sum = 0;
                double_t bot_exp_x_sum = 0;
                double_t bot_exp_nx_sum = 0;



                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    double_t top_exp_x = exp(-(top_x_max - xx) * inv_gamma * (1 + macro));
                    double_t top_exp_nx = exp(-(xx - top_x_min) * inv_gamma * (1 + macro));
                    double_t bot_exp_x = exp(-(bot_x_max - xx) * inv_gamma * (1 + macro));
                    double_t bot_exp_nx = exp(-(xx - bot_x_min) * inv_gamma * (1 + macro));

                    
                    // printf("slack: %f\n", slack);
                    // printf("inv_gamma: %f\n", inv_gamma);
                    // printf("XMAX: %f, %f\n", x_max, xx);
                    // printf("XMIN: %f, %f\n", x_min, xx);
                    // printf("XX: %f, %f\n", exp_x, xx);
                    // printf("XX_N: %f, %f\n", exp_nx, xx);

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;

                    int zz = pin_die[hyperedge_list[idx]];
                    int neg_zz = 1 - zz;

                    top_xexp_x_sum += xx * top_exp_x * zz;
                    top_xexp_nx_sum += xx * top_exp_nx * zz;
                    top_exp_x_sum += top_exp_x * zz;
                    top_exp_nx_sum += top_exp_nx * zz;
                    bot_xexp_x_sum += xx * bot_exp_x * neg_zz;
                    bot_xexp_nx_sum += xx * bot_exp_nx * neg_zz;
                    bot_exp_x_sum += bot_exp_x * neg_zz;
                    bot_exp_nx_sum += bot_exp_nx * neg_zz;
                }

                scalar_t two_layer_wl = (top_nonempty > 0 ? (top_xexp_x_sum / top_exp_x_sum - top_exp_nx_sum / top_exp_nx_sum) : 0) + (bot_nonempty > 0 ? (bot_xexp_x_sum / bot_exp_x_sum - bot_exp_nx_sum / bot_exp_nx_sum) : 0);
                scalar_t total_wl = xexp_x_sum / exp_x_sum - exp_nx_sum / exp_nx_sum;
                scalar_t wl = max(two_layer_wl, total_wl);
                partial_wa_wl[i][c] = wl;

                scalar_t b_x = inv_gamma / (exp_x_sum);
                scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
                scalar_t b_nx = -inv_gamma / (exp_nx_sum);
                scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;
                scalar_t top_b_x = inv_gamma / (top_exp_x_sum);
                scalar_t top_a_x = (1.0 - top_b_x * top_xexp_x_sum) / top_exp_x_sum;
                scalar_t top_b_nx = -inv_gamma / (top_exp_nx_sum);
                scalar_t top_a_nx = (1.0 - top_b_nx * top_xexp_nx_sum) / top_exp_nx_sum;
                scalar_t bot_b_x = inv_gamma / (bot_exp_x_sum);
                scalar_t bot_a_x = (1.0 - bot_b_x * bot_xexp_x_sum) / bot_exp_x_sum;
                scalar_t bot_b_nx = -inv_gamma / (bot_exp_nx_sum);
                scalar_t bot_a_nx = (1.0 - bot_b_nx * bot_xexp_nx_sum) / bot_exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    double_t exp_x;
                    double_t exp_nx;
                    int zz = pin_die[hyperedge_list[idx]];
                    int neg_zz = 1 - zz;

                    exp_x = exp(-(x_max - xx) * inv_gamma * (1 + macro));
                    exp_nx = exp(-(xx - x_min) * inv_gamma * (1 + macro));

                    double_t top_exp_x = exp(-(top_x_max - xx) * inv_gamma * (1 + macro));
                    double_t top_exp_nx = exp(-(xx - top_x_min) * inv_gamma * (1 + macro));
                    double_t bot_exp_x = exp(-(bot_x_max - xx) * inv_gamma * (1 + macro));
                    double_t bot_exp_nx = exp(-(xx - bot_x_min) * inv_gamma * (1 + macro));

                    if (c != 2) {
                        if (total_wl > two_layer_wl)
                            pin_grad[hyperedge_list[idx]][c] =
                                (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
                        else {
                            pin_grad[hyperedge_list[idx]][c] =
                                zz * (top_nonempty > 0 ? ((top_a_x + top_b_x * xx) * top_exp_x - (top_a_nx + top_b_nx * xx) * top_exp_nx) : 0) + 
                                neg_zz * (bot_nonempty > 0 ? ((bot_a_x + bot_b_x * xx) * bot_exp_x - (bot_a_nx + bot_b_nx * xx) * bot_exp_nx) : 0);
                        }
                        // if (pin_die[hyperedge_list[idx]] == 1)
                        //     pin_grad[hyperedge_list[idx]][c] =
                        //         (top_a_x + top_b_x * xx) * top_exp_x - (top_a_nx + top_b_nx * xx) * top_exp_nx;
                        // else if (pin_die[hyperedge_list[idx]] == 0)
                        //     pin_grad[hyperedge_list[idx]][c] =
                        //         (bot_a_x + bot_b_x * xx) * bot_exp_x - (bot_a_nx + bot_b_nx * xx) * bot_exp_nx;
                        // else
                        //     pin_grad[hyperedge_list[idx]][c] =
                        //         (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

                        // if (top_nonempty == 1) {
                        //     pin_grad[hyperedge_list[idx]][2] += 
                        //         (top_a_x + top_b_x * xx) * top_exp_x + (top_a_nx + top_b_nx * xx) * top_exp_nx;
                        // }
                        // if (bot_nonempty == 1) {
                        //     pin_grad[hyperedge_list[idx]][2] -= 
                        //         (bot_a_x + bot_b_x * xx) * bot_exp_x + (bot_a_nx + bot_b_nx * xx) * bot_exp_nx;
                        // }
                    } else {
                        pin_grad[hyperedge_list[idx]][c] +=
                            w * ((a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx);
                    }
                }
            }
        }
    }
}

__global__ void getBorder(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> border,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index / 2;  // pin index
    int c = index % 2;  // channel index
    // scan all nets, record its smallest pos, second smallest pos, second largest pos, 
    // largest pos of x,y direction respectively.
    if (i < num_nets) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        // init borders[i][0][0:2] to -inf,borders[i][2:4] to inf
        for (int die_id = 0; die_id < 2; die_id++) {
            border[i][die_id][c][0] = 3000000;
            border[i][die_id][c][1] = 3000000;
            border[i][die_id][c][2] = -10;
            border[i][die_id][c][3] = -10;
        }
        // border shape: [num_nets, 2, 2, 4]
        // detect the border of pins in bot and top die.
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = hyperedge_list[idx];
            int die_id = pin_die[pin_id];// very serious bug here, pin_die[pin_id] is not the die_id of pin_id
            if(die_id == -1) continue;
            if (pin_pos[pin_id][c] < border[i][die_id][c][0]) {
                border[i][die_id][c][1] = border[i][die_id][c][0];
                border[i][die_id][c][0] = pin_pos[pin_id][c];
            } else if (pin_pos[pin_id][c] < border[i][die_id][c][1]) {
                border[i][die_id][c][1] = pin_pos[pin_id][c];
            }
            
            if (pin_pos[pin_id][c] > border[i][die_id][c][3]) {
                border[i][die_id][c][2] = border[i][die_id][c][3];
                border[i][die_id][c][3] = pin_pos[pin_id][c];
            } else if (pin_pos[pin_id][c] > border[i][die_id][c][2]) {
                border[i][die_id][c][2] = pin_pos[pin_id][c];
            }
            if(border[i][die_id][c][2]>=0)
            {
                if((border[i][die_id][c][2]==border[i][die_id][c][3]) &&((end_idx-start_idx)==2) 
                    &&(start_idx==idx))
                {
                    int idx_tmp = idx;
                    printf("border[i][die_id][c][2] = %f, border[i][die_id][c][3] = %f, %f, %f, start_idx = %d, idx = %ld, size = %d " PRId64  "\n", border[i][die_id][c][2],
                                              border[i][die_id][c][3], pin_pos[hyperedge_list[start_idx]][c],
                                              pin_pos[hyperedge_list[start_idx+1]][c], start_idx, idx_tmp, end_idx - start_idx, idx);
                    if(idx==0)
                    {
                        printf("yes! idx ==0 !!!!!!!!");
                    }
                    assert(border[i][die_id][c][2]!=border[i][die_id][c][3]);
                }
            }
            // if((pin_pos[pin_id][c]==33351)||(pin_pos[pin_id][c]==33406)||(pin_pos[pin_id][c]==33056))
            // {
            //     printf("update!!! %f, %f, %f, %f\n", border[i][die_id][c][0], border[i][die_id][c][1],
            //                                      border[i][die_id][c][2], border[i][die_id][c][3]);
            // }
        }
    }
}


template <typename scalar_t>
__global__ void accurate_z_model_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos_invert,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> border,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // net index
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
        // const int c = index & 1;  // channel index
        scalar_t w = net_weight[i];
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        
        if (end_idx != start_idx) {
            for (int c = 0; c < 2; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);

                scalar_t xexp_x_sum = 0;
                scalar_t xexp_nx_sum = 0;
                scalar_t exp_x_sum = 0;
                scalar_t exp_nx_sum = 0;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    // if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t exp_x = exp((xx - x_max) * inv_gamma);
                    scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

                    xexp_x_sum += xx * exp_x;
                    xexp_nx_sum += xx * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;
                }

                scalar_t wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
                partial_wa_wl[i][c] = wl;

                scalar_t b_x = inv_gamma / (exp_x_sum);
                scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
                scalar_t b_nx = -inv_gamma / (exp_nx_sum);
                scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    if ((c == 2) && (pin_die[hyperedge_list[idx]] == -1)) continue; // FIXME: ignore wa_z
                    // if (pin_die[hyperedge_list[idx]] == -1) continue; // FIXME: ignore wa_z
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t exp_x = exp((xx - x_max) * inv_gamma);
                    scalar_t exp_nx = exp((x_min - xx) * inv_gamma);
                    pin_grad[hyperedge_list[idx]][c] =
                        (c == 2 ? w : 1) * ((a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx);
                }
            }
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list[idx];
                int cur_pin_die = pin_die[pin_id];
                for (int c = 0; c < 2; c++) {
                    float cur_pin_pos = pin_pos[pin_id][c];
                    if((cur_pin_pos!=border[i][cur_pin_die][c][0]) && (cur_pin_pos!=border[i][0][c][3])){
                        continue;
                    }
                    if(!(cur_pin_pos<=border[i][1 - cur_pin_die][c][3] && cur_pin_pos>=border[i][1 - cur_pin_die][c][0]))
                    {
                        continue;
                    }
                    
                    float gain=0;
                    if((cur_pin_pos==border[i][cur_pin_die][c][0]) && (border[i][cur_pin_die][c][1]<2000000)){
                        gain += (border[i][cur_pin_die][c][1] - border[i][cur_pin_die][c][0]);
                    }
                    else if((cur_pin_pos==border[i][cur_pin_die][c][3]) && (border[i][cur_pin_die][c][2]>=0)){
                        if(!(border[i][cur_pin_die][c][3]>=border[i][cur_pin_die][c][2]))
                        {
                            printf("border info: %f, %f, %f, %f, number of pins: %d, pin_id = %d, cur_pin_pos = %f, cur_pin_die = %d\n", border[i][cur_pin_die][c][0], border[i][cur_pin_die][c][1],
                                        border[i][cur_pin_die][c][2], border[i][cur_pin_die][c][3], end_idx - start_idx, pin_id, cur_pin_pos, cur_pin_die);
                            printf("inverted border info: %f, %f, %f, %f\n", border[i][1 - cur_pin_die][c][0], border[i][1 - cur_pin_die][c][1],
                                        border[i][1 - cur_pin_die][c][2], border[i][1 - cur_pin_die][c][3]);
                        }
                        assert(border[i][cur_pin_die][c][3]>=border[i][cur_pin_die][c][2]);
                        gain += (border[i][cur_pin_die][c][3] - border[i][cur_pin_die][c][2]);
                    }
                    // float pos_opposite = border[i][1 - cur_pin_die][c][0];
                    float pos_opposite = pin_pos_invert[pin_id][c];
                    if((border[i][1-cur_pin_die][c][0] < 2000000) && (pos_opposite < border[i][1-cur_pin_die][c][0]))
                    {
                        assert(border[i][1-cur_pin_die][c][3]>=border[i][1-cur_pin_die][c][0]);
                        gain -= (border[i][1-cur_pin_die][c][0] - pos_opposite);
                    }
                    else if((border[i][1-cur_pin_die][c][3] >= 0) && (pos_opposite > border[i][1-cur_pin_die][c][3]))
                    {
                        assert(border[i][1-cur_pin_die][c][3]>=border[i][1-cur_pin_die][c][0]);
                        gain -= (pos_opposite - border[i][1-cur_pin_die][c][3]);
                    }

                    float weight = 3e-5;
                    // assert((end_idx-start_idx)>2);
                    if(cur_pin_die==1)
                    {
                        pin_grad[pin_id][2] -= weight * gain;
                        // gain>0 and die ==1 --> z decrease, grad>0
                        // gain<0 and die ==1 --> z increase, grad<0
                    }
                    else{
                        pin_grad[pin_id][2] += weight * gain;
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void hpwl_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // pin index
    const int i = index;  // pin index
    if (i < num_nets) {
        // const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        for (int c = 0; c < 3; c++) {
            partial_hpwl[i][c] = 0;
            if (end_idx != start_idx) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];
                for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                }
                partial_hpwl[i][c] = abs(x_max - x_min);
            }
        }
    }
}

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    cudaSetDevice(pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nets = hyperedge_list_end.size(0);
    const int num_channels = 3;
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pos.dtype()).device(pos.device()));

    const int threads = 64;
    const int blocks = (num_nets + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(pos.scalar_type(), "hpwl_cuda", ([&] {
                              hpwl_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));

    return partial_hpwl;
}

__global__ void move_standard_cell(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> macro_list,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    int num_nodes, int num_macros) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_nodes) return;

    float xx = node_pos[index][0];
    float yy = node_pos[index][1];

    int overlap_count = 0;
    float nearest_lx = -1;
    float nearest_rx = 2000000;
    float nearest_ly = -1;
    float nearest_ry = 2000000;

    for (int i = 0; i < num_macros; i++) {
        int macro_idx = macro_list[i];
        float macro_lx = node_pos[macro_idx][0];
        float macro_ly = node_pos[macro_idx][1];
        float macro_rx = macro_lx + node_size[macro_idx][0];
        float macro_ry = macro_ly + node_size[macro_idx][1];

        // Check if the standard cell overlaps with the macro
        if (xx < macro_rx && xx > macro_lx && yy < macro_ry && yy > macro_ly) {
            overlap_count++;
            // Update nearest boundaries
            if (macro_lx > nearest_lx) nearest_lx = macro_lx;
            if (macro_rx < nearest_rx) nearest_rx = macro_rx;
            if (macro_ly > nearest_ly) nearest_ly = macro_ly;
            if (macro_ry < nearest_ry) nearest_ry = macro_ry;
        }
    }

    if (overlap_count > 1) {
        // Move the standard cell to the nearest boundary
        float dx = min(fabs(xx - nearest_lx), fabs(nearest_rx - xx));
        float dy = min(fabs(yy - nearest_ly), fabs(nearest_ry - yy));
        if (dx < dy) {
            xx = (fabs(xx - nearest_lx) < fabs(nearest_rx - xx)) ? nearest_lx : nearest_rx;
        } else {
            yy = (fabs(yy - nearest_ly) < fabs(nearest_ry - yy)) ? nearest_ly : nearest_ry;
        }

        // Set the new position of the node
        node_pos[index][0] = xx;
        node_pos[index][1] = yy;
    }
}

void force_remove_overlap_cuda(std::vector<int> macro_list, torch::Tensor& node_pos, int num_nodes, torch::Tensor node_size) {
    // Set the CUDA device to the device of node_pos
    // printf("tag2\n");
    cudaSetDevice(node_pos.get_device());

    // Get the current CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream();
    // printf("tag4\n");
    // Number of nodes
    int num_macros = macro_list.size();

    // Setup CUDA kernel execution configurations
    const int threads = 128;
    const int blocks = (num_nodes + threads - 1) / threads;

    // Create a Tensor on the same device as node_pos to hold macro_list
    torch::Tensor macro_list_cuda = torch::zeros({num_macros}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    
    // printf("tag1\n");
    // Copy data from std::vector to torch::Tensor
    auto macro_list_acc = macro_list_cuda.accessor<int, 1>();
    for (int i = 0; i < num_macros; i++) {
        macro_list_acc[i] = macro_list[i];
    }
    macro_list_cuda = macro_list_cuda.to(node_pos.device());
    // Kernel invocation
    move_standard_cell<<<blocks, threads, 0, stream>>>(
        macro_list_cuda.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nodes, num_macros
    );
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if(t != 0) {
            printf("cuda error %d\n", t);
            assert(0);
            exit(0);
        }
    }
    // TODO: Further processing or error checking

    return;
}

torch::Tensor masked_scale_hpwl_sum_cuda(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "masked_scale_hpwl_cuda", ([&] {
                              masked_scale_hpwl_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  hpwl_scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));
    const auto total_hpwl = partial_hpwl.sum();

    // return total_hpwl;
    return partial_hpwl;
}

__global__ void update_rel_cpos_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_rel_cpos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ratio_difference,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> current_node_slide_state,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // pin index
    const int i = index;  // pin index
    if (i < num_pins) {
        for (int c = 0; c < 2; c++) {
            int64_t node_id = pin_id2node_id[i];
            pin_rel_cpos[i][c] = (pin_rel_cpos[i][c] + 1e-3) * ratio_difference[i][c] - 1e-3;
            pin_rel_cpos[i][c] = pin_rel_cpos[i][c] * (1 - current_node_slide_state[node_id]) -
                                 pin_rel_cpos[i][c] * current_node_slide_state[node_id] + 1e-3;
        }
        // const int c = index & 1;  // channel index
    }
}


__global__ void get_pin_pos_and_invert_pos(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos_invert,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_rel_cpos_top,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_rel_cpos_bot,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> node_die,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int i = index >> 1;  // pin index
    const int i = index;  // pin index
    if (i < num_pins) {
        int64_t node_id = pin_id2node_id[i];
        for (int c = 0; c < 2; c++) {
            if(node_die[node_id]==0)//?????????? ==0 is top?
            {
                pin_pos[i][c] = node_pos[node_id][c] + pin_rel_cpos_top[i][c];
                pin_pos_invert[i][c] = node_pos[node_id][c] + pin_rel_cpos_bot[i][c];
            }
            else if(node_die[node_id]==1){
                pin_pos[i][c] = node_pos[node_id][c] + pin_rel_cpos_bot[i][c];
                pin_pos_invert[i][c] = node_pos[node_id][c] + pin_rel_cpos_top[i][c];
            }
        }
        pin_die[i] = node_die[node_id];
    }
}

void update_rel_cpos_cuda(torch::Tensor& pin_rel_cpos,
                     torch::Tensor pin_id2node_id,
                     torch::Tensor ratio_difference,
                     torch::Tensor current_node_slide_state)
{
    int num_pins = pin_id2node_id.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;
    update_rel_cpos_kernel<<<blocks, threads, 0, stream>>>(
            pin_rel_cpos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            ratio_difference.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            current_node_slide_state.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            num_pins);
}

__global__ void calc_node_grad_deterministic_cuda_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_grad,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list_end,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index / 3;  // node index
    if (i < num_nodes) {
        // const int c = index & 1;  // channel index
        const int c = index % 3;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = node2pin_list_end[i - 1];
        }
        int64_t end_idx = node2pin_list_end[i];
        if (end_idx != start_idx) {
            node_grad[i][c] += pin_grad[node2pin_list[start_idx]][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                node_grad[i][c] += pin_grad[node2pin_list[idx]][c];
            }
        }
    }
}

__global__ void calc_node_slide_grad_deterministic_cuda_kernel(
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_slide,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> pin_slide,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list_end,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index;  // node index
    if (i < num_nodes) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = node2pin_list_end[i - 1];
        }
        int64_t end_idx = node2pin_list_end[i];
        if (end_idx != start_idx) {
            node_slide[i] += pin_slide[node2pin_list[start_idx]];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                node_slide[i] += pin_slide[node2pin_list[idx]];
            }
        }
    }
}

void calc_node_grad_cuda(torch::Tensor node_grad,
                         torch::Tensor pin_id2node_id,
                         torch::Tensor pin_grad,
                         torch::Tensor node2pin_list,
                         torch::Tensor node2pin_list_end,
                         int num_nodes,
                         bool deterministic) {
    if (deterministic) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int threads = 128;
        const int blocks = (num_nodes * 3 + threads - 1) / threads;
        calc_node_grad_deterministic_cuda_kernel<<<blocks, threads, 0, stream>>>(
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node2pin_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node2pin_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_nodes);
    } else {
        const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 3});
        node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);
    }
}

void calc_node_slide_grad_cuda(torch::Tensor node_slide,
                         torch::Tensor pin_id2node_id,
                         torch::Tensor pin_slide,
                         torch::Tensor node2pin_list,
                         torch::Tensor node2pin_list_end,
                         int num_nodes,
                         bool deterministic) {
    if (deterministic) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int threads = 128;
        const int blocks = (num_nodes + threads - 1) / threads;
        calc_node_slide_grad_deterministic_cuda_kernel<<<blocks, threads, 0, stream>>>(
            node_slide.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            pin_slide.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            node2pin_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node2pin_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_nodes);
    } else {
        node_slide.scatter_add_(0, pin_id2node_id, pin_slide);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cuda(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor node2pin_list,
    torch::Tensor node2pin_list_end,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    torch::Tensor net_weight,
    torch::Tensor macro_mask,
    torch::Tensor ratio_multiply,
    float gamma) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_slide = torch::zeros(num_pins, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_orient = torch::zeros(num_pins, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    float inv_gamma = 1 / gamma;
    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "wa_wirelength_hpwl", ([&] {
                            //   wa_wirelength_hpwl_linear_model_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                            //   wa_wirelength_hpwl2_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  wa_wirelength_hpwl_cuda_kernel_with_pin_slide_and_pin_orient<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_rel_cpos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  net_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  partial_wa_wl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_slide.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  pin_orient.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  macro_mask.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  ratio_multiply.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  num_nets,
                                  inv_gamma);
                          }));

    // std::cout << pin_pos << std::endl;
    // std::cout << partial_hpwl << std::endl;
    // std::cout << "-----------------------------------" << std::endl;

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    auto node_slide_grad = torch::zeros(num_nodes, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    auto node_orient_grad = torch::zeros(num_nodes, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    bool deterministic = true;
    calc_node_grad_cuda(
        node_grad, pin_id2node_id, pin_grad, node2pin_list, node2pin_list_end, num_nodes, deterministic);

    calc_node_slide_grad_cuda(
        node_slide_grad, pin_id2node_id, pin_slide, node2pin_list, node2pin_list_end, num_nodes, deterministic);
    
    calc_node_slide_grad_cuda(
        node_orient_grad, pin_id2node_id, pin_orient, node2pin_list, node2pin_list_end, num_nodes, deterministic);
    // const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 3});
    // node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);

    return {partial_wa_wl, node_grad, partial_hpwl, node_slide_grad, node_orient_grad};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_accurate_hpwl_cuda(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor pin_rel_cpos_top,
    torch::Tensor pin_rel_cpos_bot,
    torch::Tensor node2pin_list,
    torch::Tensor node2pin_list_end,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    torch::Tensor net_weight,
    float gamma) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_pos_invert = pin_rel_cpos.clone();
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));


    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    // AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "get_pin_pos_and_invert_pos", ([&] {
    //                           get_pin_pos_and_invert_pos<scalar_t><<<blocks, threads, 0, stream>>>(
    //                               pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               pin_rel_cpos_top.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               pin_rel_cpos_bot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //                               pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
    //                               node_die.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
    //                               node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //                               pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
    //                               num_pins);
    //                       }));
    get_pin_pos_and_invert_pos<<<blocks, threads, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_pos_invert.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_rel_cpos_top.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_rel_cpos_bot.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        num_pins);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if(t != 0) {
            printf("cuda error %d\n", t);
            assert(0);
            exit(0);
        }
    }

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    auto border = torch::zeros({num_nets, 2, 2, 4}, torch::dtype(torch::kFloat32).device(node_pos.device()));
    getBorder<<<2 * blocks2, threads, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        border.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        num_nets);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if(t != 0) {
            printf("cuda error %d\n", t);
            assert(0);
            exit(0);
        }
    }
    float inv_gamma = 1 / gamma;

    accurate_z_model_kernel<<<blocks2, threads2, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_pos_invert.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        border.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        net_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        partial_wa_wl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        partial_hpwl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        num_nets,
        inv_gamma);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if(t != 0) {
            printf("cuda error %d\n", t);
            assert(0);
            exit(0);
        }
    }
    // std::cout << pin_pos << std::endl;
    // std::cout << partial_hpwl << std::endl;
    // std::cout << "-----------------------------------" << std::endl;

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    bool deterministic = true;
    calc_node_grad_cuda(
        node_grad, pin_id2node_id, pin_grad, node2pin_list, node2pin_list_end, num_nodes, deterministic);
    // const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 3});
    // node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);

    return {partial_wa_wl, node_grad, partial_hpwl};
}

}  // namespace GP3D