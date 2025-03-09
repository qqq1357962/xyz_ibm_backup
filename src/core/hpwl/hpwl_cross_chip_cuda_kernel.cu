#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "hpwl_cuda.cuh"

template <typename scalar_t>
__global__ void node_pos_to_pin_pos_cross_chip_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> node_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die, int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // pin index
    if (i < num_pins) {
        const int c = index & 1;  // channel index
        int64_t node_id = pin_id2node_id[i];

        pin_pos[i][c] += node_pos[node_id][c];
        pin_die[i] = node_die[node_id];
    }
}  // END MODULE

template <typename scalar_t>
__global__ void hpwl_cross_chip_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_ovlp_hpwl, int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;     // pin index
    if (i < num_nets) {           // TODO:  && net_mask[i]
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        partial_hpwl[0][i][c] = 0;
        partial_hpwl[1][i][c] = 0;

        int64_t bot_count = 0;
        int64_t top_count = 0;
        scalar_t x_min_bot = 0;
        scalar_t x_max_bot = 0;
        scalar_t x_min_top = 0;
        scalar_t x_max_top = 0;

        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
            scalar_t xx = pin_pos[pin_id][c];
            if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                if (bot_count == 0) {
                    x_min_bot = xx;
                    x_max_bot = xx;
                } else {
                    x_min_bot = min(xx, x_min_bot);
                    x_max_bot = max(xx, x_max_bot);
                }
                bot_count++;
            }
            if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
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
        partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
        partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);

        scalar_t x_max_mid = min(x_max_bot, x_max_top);
        scalar_t x_min_mid = max(x_min_bot, x_min_top);
        partial_ovlp_hpwl[i][c] = max(x_max_mid - x_min_mid, (scalar_t)0);
    }
}  // END MODULE

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hpwl_cross_chip_cuda(
    torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_ovlp_hpwl =
        torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    // TODO: threads
    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl_cross_chip", ([&] {
                              hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  partial_ovlp_hpwl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));

    // return {torch::sum((partial_hpwl.index({0, "..."})).sum(1)),
    //         torch::sum((partial_hpwl.index({1, "..."})).sum(1))};  // TODO:
    // return {(partial_hpwl.index({0, "..."})).sum(2), (partial_hpwl.index({1, "..."})).sum(2)};  // TODO:

    return {(partial_hpwl.index({0, "..."})).sum(1), (partial_hpwl.index({1, "..."})).sum(1), partial_ovlp_hpwl.sum(1)};  // TODO:

}  // END MODULE

template <typename scalar_t>
__global__ void masked_scale_hpwl_cross_chip_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hpwl_scale,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // net index
    if (i < num_nets && net_mask[i]) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        partial_hpwl[0][i][c] = 0;
        partial_hpwl[1][i][c] = 0;

        int64_t bot_count = 0;
        int64_t top_count = 0;
        scalar_t x_min_bot = 0;
        scalar_t x_max_bot = 0;
        scalar_t x_min_top = 0;
        scalar_t x_max_top = 0;

        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
            scalar_t xx = pin_pos[pin_id][c];
            if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                if (bot_count == 0) {
                    x_min_bot = xx;
                    x_max_bot = xx;
                } else {
                    x_min_bot = min(xx, x_min_bot);
                    x_max_bot = max(xx, x_max_bot);
                }
                bot_count++;
            }
            if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
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
        partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
        partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
        partial_hpwl[0][i][c] = roundf(abs(x_max_bot - x_min_bot) * hpwl_scale[c]);
        partial_hpwl[1][i][c] = roundf(abs(x_max_top - x_min_top) * hpwl_scale[c]);
    }
}


torch::Tensor masked_scale_hpwl_sum_cross_chip_cuda(torch::Tensor node_pos,
                                         torch::Tensor node_die,
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
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    const int threads2 = 128;
    const int blocks2 = (num_nets * 2 + threads2 - 1) / threads2;

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "masked_scale_hpwl_cuda", ([&] {
                              masked_scale_hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  hpwl_scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));
    const auto total_hpwl = partial_hpwl.sum();

    return total_hpwl;  // FIXME:
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hpwl_formatted_cuda(
    torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    pin_pos = torch::_cast_Int(pin_pos);
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_ovlp_hpwl =
        torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl_cross_chip", ([&] {
                              hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  partial_ovlp_hpwl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));
    // TODO: hardcode die_scale & site_width
    return {torch::sum((partial_hpwl.index({0, "..."})).sum(1)), torch::sum((partial_hpwl.index({1, "..."})).sum(1)),
            torch::sum(partial_ovlp_hpwl.sum(1))};
}  // END MODULE

template <typename scalar_t>
__global__ void wa_wirelength_hpwl_cross_chip_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_grad, int num_nets, float inv_gamma) {
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
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    x_min = min(xx, x_min);
                    x_max = max(xx, x_max);
                    if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                        if (bot_count == 0) {
                            x_min_bot = xx;
                            x_max_bot = xx;
                        } else {
                            x_min_bot = min(xx, x_min_bot);
                            x_max_bot = max(xx, x_max_bot);
                        }
                        bot_count++;
                    }
                    if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
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
                // partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
                // partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
                scalar_t x_max_mid = min(x_max_bot, x_max_top);
                scalar_t x_min_mid = max(x_min_bot, x_min_top);
                partial_hpwl[0][i][c] = max(x_max_mid - x_min_mid, (scalar_t)0);

                scalar_t xexp_x_sum_bot = 0;
                scalar_t xexp_nx_sum_bot = 0;
                scalar_t exp_x_sum_bot = 0;
                scalar_t exp_nx_sum_bot = 0;

                scalar_t xexp_x_sum_top = 0;
                scalar_t xexp_nx_sum_top = 0;
                scalar_t exp_x_sum_top = 0;
                scalar_t exp_nx_sum_top = 0;

                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                        scalar_t exp_x = exp((xx - x_max_bot) * inv_gamma);
                        scalar_t exp_nx = exp((x_min_bot - xx) * inv_gamma);

                        xexp_x_sum_bot += xx * exp_x;
                        xexp_nx_sum_bot += xx * exp_nx;
                        exp_x_sum_bot += exp_x;
                        exp_nx_sum_bot += exp_nx;
                    }
                    if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {
                        scalar_t exp_x = exp((xx - x_max_top) * inv_gamma);
                        scalar_t exp_nx = exp((x_min_top - xx) * inv_gamma);

                        xexp_x_sum_top += xx * exp_x;
                        xexp_nx_sum_top += xx * exp_nx;
                        exp_x_sum_top += exp_x;
                        exp_nx_sum_top += exp_nx;
                    }
                }

                scalar_t wl_bot = xexp_x_sum_bot / exp_x_sum_bot - xexp_nx_sum_bot / exp_nx_sum_bot;
                scalar_t wl_top = xexp_x_sum_top / exp_x_sum_top - xexp_nx_sum_top / exp_nx_sum_top;
                partial_wa_wl[0][i][c] = exp_x_sum_bot == 0 ? 0 : wl_bot;
                partial_wa_wl[1][i][c] = exp_x_sum_top == 0 ? 0 : wl_top;

                scalar_t b_x_bot = inv_gamma / (exp_x_sum_bot);
                scalar_t a_x_bot = (1.0 - b_x_bot * xexp_x_sum_bot) / exp_x_sum_bot;
                scalar_t b_nx_bot = -inv_gamma / (exp_nx_sum_bot);
                scalar_t a_nx_bot = (1.0 - b_nx_bot * xexp_nx_sum_bot) / exp_nx_sum_bot;
                scalar_t b_x_top = inv_gamma / (exp_x_sum_top);
                scalar_t a_x_top = (1.0 - b_x_top * xexp_x_sum_top) / exp_x_sum_top;
                scalar_t b_nx_top = -inv_gamma / (exp_nx_sum_top);
                scalar_t a_nx_top = (1.0 - b_nx_top * xexp_nx_sum_top) / exp_nx_sum_top;
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list[idx];
                    scalar_t xx = pin_pos[pin_id][c];
                    if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                        scalar_t exp_x = exp((xx - x_max_bot) * inv_gamma);
                        scalar_t exp_nx = exp((x_min_bot - xx) * inv_gamma);

                        pin_grad[pin_id][c] += (a_x_bot + b_x_bot * xx) * exp_x - (a_nx_bot + b_nx_bot * xx) * exp_nx;
                    }
                    if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {
                        scalar_t exp_x = exp((xx - x_max_top) * inv_gamma);
                        scalar_t exp_nx = exp((x_min_top - xx) * inv_gamma);

                        pin_grad[pin_id][c] += (a_x_top + b_x_top * xx) * exp_x - (a_nx_top + b_nx_top * xx) * exp_nx;
                    }
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cross_chip_cuda(
    torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end, torch::Tensor net_mask, float gamma, bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(pin_pos.device()));
    auto partial_wa_wl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

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
                              wa_wirelength_hpwl_cross_chip_cuda_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
                                  partial_wa_wl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  pin_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), num_nets,
                                  inv_gamma);
                          }));

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, num_channels});
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
                    if (pin_die[pin_id] == 0) {
                        x_min_bot = (bot_count == 0) ? xx : min(xx, x_min_bot);
                        x_max_bot = (bot_count == 0) ? xx : max(xx, x_max_bot);
                        bot_count++;
                    }
                    if (pin_die[pin_id] == 1) {
                        x_min_top = (top_count == 0) ? xx : min(xx, x_min_top);
                        x_max_top = (top_count == 0) ? xx : max(xx, x_max_top);
                        top_count++;
                    }
                }

                if (bot_count == 0) x_max_bot = x_max;
                if (top_count == 0) x_max_top = x_max;
                if (bot_count == 0) x_min_bot = x_min;
                if (top_count == 0) x_min_top = x_min;

                scalar_t x_max_mid = min(x_max_bot, x_max_top);
                scalar_t x_min_mid = max(x_min_bot, x_min_top);
                scalar_t x_ovlp = max(x_max_mid - x_min_mid, (scalar_t)0);
                partial_hpwl[0][i][c] = x_ovlp;
                if (x_ovlp != 0) {
                    partial_hpwl[1][i][c] = 1;
                }


                via_pos[i][c] = (x_max_mid + x_min_mid) / 2;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> place_via_to_optim_cuda(torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id,
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
    const int blocks = (num_pins * 2 + threads - 1) / threads;

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

    return {via_pos, partial_hpwl};
}

template <typename scalar_t>
__global__ void evaluate_via_from_optim_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_die,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> partial_hpwl, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> partial_hpwl_overhead, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> hpwl_optim_bbox, 
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;     // pin index
    if (i < num_nets) {           // TODO:  && net_mask[i]
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        partial_hpwl[0][i][c] = 0;
        partial_hpwl[1][i][c] = 0;

        int64_t bot_count = 0;
        int64_t top_count = 0;
        scalar_t x_min_bot = 0;
        scalar_t x_max_bot = 0;
        scalar_t x_min_top = 0;
        scalar_t x_max_top = 0;

        scalar_t x_min_bot_wo_via = std::numeric_limits<scalar_t>::max();
        scalar_t x_max_bot_wo_via = 0;
        scalar_t x_min_top_wo_via = std::numeric_limits<scalar_t>::max();
        scalar_t x_max_top_wo_via = 0;

        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
            scalar_t xx = pin_pos[pin_id][c];
            if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
                if (bot_count == 0) {
                    x_min_bot = xx;
                    x_max_bot = xx;
                } else {
                    x_min_bot = min(xx, x_min_bot);
                    x_max_bot = max(xx, x_max_bot);
                }
                bot_count++;
            }
            if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
                if (top_count == 0) {
                    x_min_top = xx;
                    x_max_top = xx;
                } else {
                    x_min_top = min(xx, x_min_top);
                    x_max_top = max(xx, x_max_top);
                }
                top_count++;
            }
            if (pin_die[pin_id] == 0) {
                x_min_bot_wo_via = min(xx, x_min_bot_wo_via);
                x_max_bot_wo_via = max(xx, x_max_bot_wo_via);
            }
            if (pin_die[pin_id] == 1) {
                x_min_top_wo_via = min(xx, x_min_top_wo_via);
                x_max_top_wo_via = max(xx, x_max_top_wo_via);
            }
        }
        partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
        partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);

        scalar_t x_max_mid_wo_via = min(x_max_bot_wo_via, x_max_top_wo_via);
        scalar_t x_min_mid_wo_via = max(x_min_bot_wo_via, x_min_top_wo_via);
        scalar_t x_max_wo_via = max(x_max_bot_wo_via, x_max_top_wo_via);
        scalar_t x_min_wo_via = min(x_min_bot_wo_via, x_min_top_wo_via);
        scalar_t expected_len = (x_max_wo_via - x_min_wo_via) + max(x_max_mid_wo_via - x_min_mid_wo_via, (scalar_t)0);
        scalar_t actual_len = partial_hpwl[0][i][c] + partial_hpwl[1][i][c];

        if (top_count * bot_count) {
            hpwl_optim_bbox[i][c] = x_min_mid_wo_via;
            hpwl_optim_bbox[i][2 + c] = x_max_mid_wo_via;
            hpwl_optim_bbox[i][4 + c] = x_min_wo_via;
            hpwl_optim_bbox[i][6 + c] = x_max_wo_via;
        }

        // partial_ovlp_hpwl[i][c] = max(x_max_mid - x_min_mid, (scalar_t)0);
        partial_hpwl_overhead[i][c] = actual_len - expected_len;
    }
}  // END MODULE

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_via_from_optim_cuda(
    torch::Tensor node_pos, torch::Tensor node_die, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt).device(node_pos.device()));
    auto partial_hpwl =
        torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl_overhead =
        torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    /* TODO: net optim bbox */
    auto hpwl_optim_bbox =
        torch::zeros({num_nets, num_channels * 4}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                              node_pos_to_pin_pos_cross_chip_cuda_forward_kernel<scalar_t>
                                  <<<blocks, threads, 0, stream>>>(
                                      node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      node_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                      pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), num_pins);
                          }));

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "evaluate_via_from_optim", ([&] {
                              evaluate_via_from_optim_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_die.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  partial_hpwl.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  partial_hpwl_overhead.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  hpwl_optim_bbox.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nets);
                          }));
    
    return {(partial_hpwl.index({0, "..."})), (partial_hpwl.index({1, "..."})), partial_hpwl_overhead, hpwl_optim_bbox};

}  // END MODULE