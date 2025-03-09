#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "hpwl_cuda.cuh"

namespace GP3D {

template <typename scalar_t>
__global__ void node_pos_to_pin_pos_cross_chip_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> node_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die, int num_pins) {
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
__global__ void masked_scale_hpwl_cross_chip_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hpwl_scale,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl, int num_nets) {
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
            for (int c = 0; c < 2; c++) {
                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];

                int64_t bot_count = 0;
                int64_t top_count = 0;
                scalar_t x_min_bot = 0;
                scalar_t x_max_bot = 0;
                scalar_t x_min_top = 0;
                scalar_t x_max_top = 0;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                    int64_t pin_id = hyperedge_list[idx];
                    if (pin_die[pin_id] == 0) {
                        if (bot_count == 0) {
                            x_min_bot = xx;
                            x_max_bot = xx;
                        } else {
                            x_min_bot = min(xx, x_min_bot);
                            x_max_bot = max(xx, x_max_bot);
                        }
                        bot_count++;
                    }
                    if (pin_die[pin_id] == 1) {  // FIXME: fix bug: if <- else if
                        if (top_count == 0) {
                            x_min_top = xx;
                            x_max_top = xx;
                        } else {
                            x_min_top = min(xx, x_min_top);
                            x_max_top = max(xx, x_max_top);
                        }
                        top_count++;
                    }
                }

                // partial_hpwl[0][i][c] = roundf(abs(x_max_bot - x_min_bot) * hpwl_scale[c]);
                // partial_hpwl[1][i][c] = roundf(abs(x_max_top - x_min_top) * hpwl_scale[c]);
                // if ((abs(x_max_bot - x_min_bot) + abs(x_max_top - x_min_top)) < abs(x_max - x_min)) {
                //     partial_hpwl[0][i][c] = abs(x_max - x_min);
                // } else {
                partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
                partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
                // }
            }
        }
    }
}

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_ovlp_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> die_info,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> net_weight,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> node_optim_info, int num_nets, float inv_gamma,
    bool correlate_bbox, float bbox_correlation) {
    scalar_t sig_coef = 10;
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
                scalar_t xx_ur = die_info[2 * c + 1];

                int64_t pin_id = hyperedge_list[start_idx];
                scalar_t x_min = pin_pos[pin_id][c];
                scalar_t x_max = pin_pos[pin_id][c];

                int64_t bot_count = 0;
                int64_t top_count = 0;
                scalar_t x_min_bot = 0;
                scalar_t x_max_bot = 0;
                scalar_t x_min_top = 0;
                scalar_t x_max_top = 0;

                /* collect min/max box */
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                    if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                        x_min_bot = bot_count == 0 ? xx : min(xx, x_min_bot);
                        x_max_bot = bot_count == 0 ? xx : max(xx, x_max_bot);
                        bot_count++;
                    }
                    if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
                        x_min_top = top_count == 0 ? xx : min(xx, x_min_top);
                        x_max_top = top_count == 0 ? xx : max(xx, x_max_top);
                        top_count++;
                    }
                }

                /* return hpwl info */
                scalar_t x_max_mid = min(x_max_bot, x_max_top);
                scalar_t x_min_mid = max(x_min_bot, x_min_top);
                scalar_t x_ovlp = max(x_max_mid - x_min_mid, (scalar_t)0);
                partial_hpwl[0][i][c] = x_ovlp;
                x_max_mid = x_max_mid == 0 ? max(x_max_bot, x_max_top) : x_max_mid;
                scalar_t x_mid = (x_max_mid + x_min_mid) / 2;

                // partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
                // partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
                if (x_ovlp == 0) {
                    for (int64_t idx = start_idx; idx < end_idx; idx++) {
                        int64_t pin_id = hyperedge_list[idx];
                        int64_t node_id = pin_id2node_id[pin_id];
                        node_optim_info[node_id] = 1;
                    }
                }

                /* apply grad on x/y */
                scalar_t xexp_x_sum = 0;
                scalar_t xexp_nx_sum = 0;
                scalar_t exp_x_sum = 0;
                scalar_t exp_nx_sum = 0;

                scalar_t exp_x_max = 0;
                scalar_t exp_nx_max = 0;

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

                    exp_x_max = max(exp_x_max, exp_x);
                    exp_nx_max = max(exp_nx_max, exp_nx);
                }
                // if (threadIdx.x == 0) {
                //     printf("%.2f, %.2f =  %.2f\n", exp_x_max, exp_nx_max, xexp_x_sum / exp_x_sum);
                // }

                scalar_t wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
                partial_hpwl[0][i][c] = wl;
                // partial_hpwl[0][i][c] = abs(x_max - x_min);

                scalar_t b_x = inv_gamma / (exp_x_sum);
                scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
                scalar_t b_nx = -inv_gamma / (exp_nx_sum);
                scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[hyperedge_list[idx]][c];
                    scalar_t exp_x = exp((xx - x_max) * inv_gamma);
                    scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

                    pin_grad[pin_id][c] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
                    // pin_grad[pin_id][2] = 0;
                }
                /* wa hpwl on z */
                scalar_t xexp_x_sum_bot = 0;
                scalar_t xexp_nx_sum_bot = 0;
                scalar_t exp_x_sum_bot = 0;
                scalar_t exp_nx_sum_bot = 0;
                scalar_t xexp_x_sum_top = 0;
                scalar_t xexp_nx_sum_top = 0;
                scalar_t exp_x_sum_top = 0;
                scalar_t exp_nx_sum_top = 0;
                scalar_t x_min_neg = xx_ur - x_min;
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    scalar_t xx_neg = xx_ur - xx;
                    scalar_t zz = pin_pos[pin_id][2];
                    scalar_t zz_neg = 1 - zz;

                    // TODO: SIGMOID
                    // zz = 1 / (1 + exp(-(zz - 0.5) * sig_coef));
                    // zz_neg = 1 / (1 + exp(-(zz_neg - 0.5) * sig_coef));

                    scalar_t exp_x = exp((zz * xx - x_max) * inv_gamma);
                    // scalar_t exp_nx = exp((x_min - zz * xx) * inv_gamma);
                    scalar_t exp_nx = exp((zz * xx_neg - x_min_neg) * inv_gamma);
                    xexp_x_sum_top += xx * exp_x; // FIXME:
                    xexp_nx_sum_top += xx_neg * exp_nx;
                    exp_x_sum_top += exp_x;
                    exp_nx_sum_top += exp_nx;

                    exp_x = exp((zz_neg * xx - x_max) * inv_gamma);
                    // exp_nx = exp((x_min - zz_neg * xx) * inv_gamma);
                    exp_nx = exp((zz_neg * xx_neg - x_min_neg) * inv_gamma);
                    xexp_x_sum_bot += xx * exp_x;
                    xexp_nx_sum_bot += xx_neg * exp_nx;
                    exp_x_sum_bot += exp_x;
                    exp_nx_sum_bot += exp_nx;
                }

                /* correlate bbox */
                scalar_t x_correlation = 1;
                if (correlate_bbox) {
                    scalar_t x_max0 = xexp_x_sum_bot / exp_x_sum_bot;
                    scalar_t x_min0 = xx_ur - (xexp_nx_sum_bot / exp_nx_sum_bot);
                    scalar_t x_max1 = xexp_x_sum_top / exp_x_sum_top;
                    scalar_t x_min1 = xx_ur - (xexp_nx_sum_top / exp_nx_sum_top);
                    scalar_t x_correlation =
                        max((scalar_t)(exp(((x_max1 - x_min0) * (x_max0 - x_min1)) / bbox_correlation)), (scalar_t)1);
                }

                scalar_t wl_bot = xexp_x_sum_bot / exp_x_sum_bot + xexp_nx_sum_bot / exp_nx_sum_bot - xx_ur;
                scalar_t wl_top = xexp_x_sum_top / exp_x_sum_top + xexp_nx_sum_top / exp_nx_sum_top - xx_ur;
                if (wl_bot < 0) wl_bot = 0;
                if (wl_top < 0) wl_top = 0;
                // partial_wa_wl[0][i][c] = bot_count == 0 ? 0 : wl_bot;
                // partial_wa_wl[1][i][c] = top_count == 0 ? 0 : wl_top;
                partial_wa_wl[0][i][c] = wl_bot;
                partial_wa_wl[1][i][c] = wl_top;

                // if (threadIdx.x == 0) {
                //     // printf("net %d ", i);
                //     // for (int64_t idx = start_idx; idx < end_idx; idx++) {
                //     //     int64_t pin_id = hyperedge_list[idx];
                //     //     scalar_t xx = pin_pos[pin_id][c];
                //     //     scalar_t zz = pin_pos[pin_id][2];
                //     //     printf("%.2E at %.2E | ", xx, zz);
                //     // }
                //     if ((end_idx - start_idx) == 3) {
                //         printf("net %d ++ %.2E at %.2E | %.2E at %.2E | %.2E at %.2E \n net-%d %.2E - %.2E / %.2E - %.2E / %.2E - %.2E  %.3f\n", i, 
                //             pin_pos[hyperedge_list[start_idx]][c], pin_pos[hyperedge_list[start_idx]][2],
                //             pin_pos[hyperedge_list[start_idx + 1]][c], pin_pos[hyperedge_list[start_idx + 1]][2],
                //             pin_pos[hyperedge_list[start_idx + 2]][c], pin_pos[hyperedge_list[start_idx + 2]][2], i, 
                //             xexp_x_sum_bot / exp_x_sum_bot, -(xexp_nx_sum_bot / exp_nx_sum_bot - xx_ur),
                //             xexp_x_sum_top / exp_x_sum_top, -(xexp_nx_sum_top / exp_nx_sum_top - xx_ur),
                //             xexp_x_sum / exp_x_sum, (xexp_nx_sum / exp_nx_sum)),
                //             inv_gamma;
                //     }
                //     // printf("net-%d %.2E - %.2E / %.2E - %.2E / %.2E - %.2E\n", i, 
                //     //     xexp_x_sum_bot / exp_x_sum_bot, -(xexp_nx_sum_bot / exp_nx_sum_bot - xx_ur),
                //     //     xexp_x_sum_top / exp_x_sum_top, -(xexp_nx_sum_top / exp_nx_sum_top - xx_ur),
                //     //     xexp_x_sum / exp_x_sum, (xexp_nx_sum / exp_nx_sum ));
                // }

                /* grad z */
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    scalar_t xx_neg = xx_ur - xx;
                    scalar_t x_min_neg = xx_ur - x_min;
                    scalar_t zz = pin_pos[pin_id][2];
                    scalar_t zz_neg = 1 - zz;

                    // TODO: SIGMOID
                    zz = 1 / (1 + exp(-(zz - 0.5) * sig_coef));
                    zz_neg = 1 / (1 + exp(-(zz_neg - 0.5) * sig_coef));
                    scalar_t grad_sig_z = zz * (1 - zz);
                    scalar_t grad_sig_z_neg = zz_neg * (1 - zz_neg);
                    scalar_t exp_x;
                    scalar_t exp_nx;
                    scalar_t b_x;
                    scalar_t a_x;
                    scalar_t b_nx;
                    scalar_t a_nx;

                    if (top_count > 0) {
                        exp_x = exp((zz * xx - x_max) * inv_gamma);
                        exp_nx = exp((zz * xx_neg - x_min_neg) * inv_gamma);
                        b_x = xx * inv_gamma / (exp_x_sum_top);
                        a_x = (xx - b_x * xexp_x_sum_top) / exp_x_sum_top;
                        b_nx = xx_neg * inv_gamma / (exp_nx_sum_top);
                        a_nx = (xx_neg - b_nx * xexp_nx_sum_top) / exp_nx_sum_top;
                        pin_grad[pin_id][2] += ((a_x + b_x * xx * zz) * exp_x + (a_nx + b_nx * xx_neg * zz) * exp_nx) *
                                            grad_sig_z * sig_coef * w * x_correlation;
                    }
                    
                    if (bot_count > 0) {
                        exp_x = exp((zz_neg * xx - x_max) * inv_gamma);
                        exp_nx = exp((zz_neg * xx_neg - x_min_neg) * inv_gamma);
                        b_x = xx * inv_gamma / (exp_x_sum_bot);
                        a_x = (xx - b_x * xexp_x_sum_bot) / exp_x_sum_bot;
                        b_nx = xx_neg * inv_gamma / (exp_nx_sum_bot);
                        a_nx = (xx_neg - b_nx * xexp_nx_sum_bot) / exp_nx_sum_bot;
                        pin_grad[pin_id][2] -=
                            ((a_x + b_x * xx * zz_neg) * exp_x + (a_nx + b_nx * xx_neg * zz_neg) * exp_nx) *
                            grad_sig_z_neg * sig_coef * w * x_correlation;
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void hpwl_cross_chip_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl, int num_nets) {
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

        for (int c = 0; c < 2; c++) {
            int64_t bot_count = 0;
            int64_t top_count = 0;
            scalar_t x_min_bot = 0;
            scalar_t x_max_bot = 0;
            scalar_t x_min_top = 0;
            scalar_t x_max_top = 0;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
                if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                    if (bot_count == 0) {
                        x_min_bot = pin_pos[pin_id][c];
                        x_max_bot = pin_pos[pin_id][c];
                    } else {
                        scalar_t xx = pin_pos[pin_id][c];
                        x_min_bot = min(xx, x_min_bot);
                        x_max_bot = max(xx, x_max_bot);
                    }
                    bot_count++;
                }
                if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
                    if (top_count == 0) {
                        x_min_top = pin_pos[pin_id][c];
                        x_max_top = pin_pos[pin_id][c];
                    } else {
                        scalar_t xx = pin_pos[pin_id][c];
                        x_min_top = min(xx, x_min_top);
                        x_max_top = max(xx, x_max_top);
                    }
                    top_count++;
                }
            }
            partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
            partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> hpwl_cross_chip_cuda(torch::Tensor node_pos, torch::Tensor node_die,
                                                              torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
                                                              torch::Tensor hyperedge_list,
                                                              torch::Tensor hyperedge_list_end) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(node_pos.dtype()).device(node_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "hpwl_cuda", ([&] {
                              hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), num_nets);
                          }));

    // return partial_hpwl;
    return {(partial_hpwl.index({0, "..."})), (partial_hpwl.index({1, "..."}))};  // TODO:
}

torch::Tensor masked_scale_hpwl_cross_chip_cuda(torch::Tensor node_pos, torch::Tensor node_die,
                                                torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
                                                torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end,
                                                torch::Tensor net_mask, torch::Tensor hpwl_scale) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(pin_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "masked_scale_hpwl_cuda", ([&] {
                              masked_scale_hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  hpwl_scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), num_nets);
                          }));
    const auto total_hpwl = partial_hpwl.sum();

    // return total_hpwl;
    return partial_hpwl;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_ovlp_cuda(
    torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor die_info, torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end, torch::Tensor net_mask,
    torch::Tensor net_weight, torch::Tensor node_optim_info, float gamma, bool correlate_bbox, float bbox_correlation) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 3;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(pin_pos.device()));
    auto partial_wa_wl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    float inv_gamma = 1 / gamma;
    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "wa_wirelength_hpwl_ovl", ([&] {
                              wa_wirelength_hpwl_ovlp_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  die_info.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  net_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  partial_wa_wl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  pin_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  node_optim_info.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_nets,
                                  inv_gamma, correlate_bbox, bbox_correlation);
                          }));

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 3});
    node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);

    return {partial_wa_wl, node_grad, partial_hpwl};
}

template <typename scalar_t>
__global__ void place_via_to_optim_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> via_pos, int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index;  // pin index
    if (i < num_nets && net_mask[i]) {
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

                int64_t bot_count = 0;
                int64_t top_count = 0;
                scalar_t x_min_bot = 0;
                scalar_t x_max_bot = 0;
                scalar_t x_min_top = 0;
                scalar_t x_max_top = 0;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                    if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                        x_min_bot = bot_count == 0 ? xx : min(xx, x_min_bot);
                        x_max_bot = bot_count == 0 ? xx : max(xx, x_max_bot);
                        bot_count++;
                    }
                    if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
                        x_min_top = top_count == 0 ? xx : min(xx, x_min_top);
                        x_max_top = top_count == 0 ? xx : max(xx, x_max_top);
                        top_count++;
                    }
                }

                scalar_t x_max_mid = min(x_max_bot, x_max_top);
                scalar_t x_min_mid = max(x_min_bot, x_min_top);
                partial_hpwl[0][i][c] = max(x_max_mid - x_min_mid, (scalar_t)0);
                via_pos[i][c] = (x_max_mid + x_min_mid) / 2;
            }
        }
    }
}

torch::Tensor place_via_to_optim_cuda(torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id,
                                      torch::Tensor pin_rel_cpos, torch::Tensor hyperedge_list,
                                      torch::Tensor hyperedge_list_end, torch::Tensor net_mask) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(pin_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto via_pos = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets + threads2 - 1) / threads2;

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "place_via_to_optim_cuda_kernel", ([&] {
                              place_via_to_optim_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  via_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), num_nets);
                          }));

    return via_pos;
}

}  // namespace GP3D