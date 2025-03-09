// #include "hpwl.h"

// namespace wa_wirelength_hpwl {

// template <typename scalar_t>
// void node_pos_to_pin_pos_forward_kernel(const torch::TensorAccessor<scalar_t, 2> node_pos,
//                                         const torch::TensorAccessor<int64_t, 1> pin_id2node_id,
//                                         torch::TensorAccessor<scalar_t, 2> pin_pos,
//                                         int num_pins,
//                                         const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_pins; ++i) {
//         for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//             int64_t node_id = pin_id2node_id[i];
//             pin_pos[i][c] += node_pos[node_id][c];  // TODO: atomic_add
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void masked_scale_hpwl_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
//                               const torch::TensorAccessor<int64_t, 1> hyperedge_list,
//                               const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
//                               const torch::TensorAccessor<bool, 1> net_mask,
//                               const torch::TensorAccessor<scalar_t, 1> hpwl_scale,
//                               torch::TensorAccessor<scalar_t, 2> partial_hpwl,
//                               int num_nets,
//                               const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nets; ++i) {
//         if (net_mask[i]) {
//             for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//                 int64_t start_idx = 0;
//                 if (i != 0) {
//                     start_idx = hyperedge_list_end[i - 1];
//                 }
//                 int64_t end_idx = hyperedge_list_end[i];
//                 if (end_idx != start_idx) {
//                     int64_t pin_id = hyperedge_list[start_idx];
//                     scalar_t x_min = pin_pos[pin_id][c];
//                     scalar_t x_max = pin_pos[pin_id][c];
//                     for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
//                         scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                         x_min = min(xx, x_min);
//                         x_max = max(xx, x_max);
//                     }
//                     partial_hpwl[i][c] = round(abs(x_max - x_min) * hpwl_scale[c]);
//                 }
//             }
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void wa_wirelength_hpwl_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
//                                const torch::TensorAccessor<int64_t, 1> hyperedge_list,
//                                const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
//                                const torch::TensorAccessor<bool, 1> net_mask,
//                                torch::TensorAccessor<scalar_t, 2> partial_wa_wl,
//                                torch::TensorAccessor<scalar_t, 2> partial_hpwl,
//                                torch::TensorAccessor<scalar_t, 2> pin_grad,
//                                int num_nets,
//                                float inv_gamma,
//                                const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nets; ++i) {
//         if (net_mask[i]) {
//             for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//                 int64_t start_idx = 0;
//                 if (i != 0) {
//                     start_idx = hyperedge_list_end[i - 1];
//                 }
//                 int64_t end_idx = hyperedge_list_end[i];
//                 if (end_idx != start_idx) {
//                     int64_t pin_id = hyperedge_list[start_idx];
//                     scalar_t x_min = pin_pos[pin_id][c];
//                     scalar_t x_max = pin_pos[pin_id][c];
//                     for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
//                         scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                         x_min = min(xx, x_min);
//                         x_max = max(xx, x_max);
//                     }
//                     partial_hpwl[i][c] = abs(x_max - x_min);

//                     scalar_t xexp_x_sum = 0;
//                     scalar_t xexp_nx_sum = 0;
//                     scalar_t exp_x_sum = 0;
//                     scalar_t exp_nx_sum = 0;

//                     for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                         scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                         scalar_t exp_x = exp((xx - x_max) * inv_gamma);
//                         scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

//                         xexp_x_sum += xx * exp_x;
//                         xexp_nx_sum += xx * exp_nx;
//                         exp_x_sum += exp_x;
//                         exp_nx_sum += exp_nx;
//                     }

//                     scalar_t wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
//                     partial_wa_wl[i][c] = wl;

//                     scalar_t b_x = inv_gamma / (exp_x_sum);
//                     scalar_t a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
//                     scalar_t b_nx = -inv_gamma / (exp_nx_sum);
//                     scalar_t a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

//                     for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                         scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                         scalar_t exp_x = exp((xx - x_max) * inv_gamma);
//                         scalar_t exp_nx = exp((x_min - xx) * inv_gamma);

//                         pin_grad[hyperedge_list[idx]][c] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
//                     }
//                 }
//             }
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl(torch::Tensor node_pos,
//                                                                                      torch::Tensor pin_id2node_id,
//                                                                                      torch::Tensor pin_rel_cpos,
//                                                                                      torch::Tensor hyperedge_list,
//                                                                                      torch::Tensor
//                                                                                      hyperedge_list_end,
//                                                                                      torch::Tensor net_mask,
//                                                                                      float gamma) {
//     CHECK_INPUT(node_pos);

//     const auto num_nodes = node_pos.size(0);
//     const auto num_pins = pin_id2node_id.size(0);  // FIXME:
//     const auto num_nets = hyperedge_list_end.size(0);
//     const auto num_channels = 2;  // x, y

//     auto pin_pos = pin_rel_cpos.clone();  // pin
//     auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));
//     auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));
//     auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_forward", ([&] {
//                                    node_pos_to_pin_pos_forward_kernel<scalar_t>(node_pos.accessor<scalar_t, 2>(),
//                                                                                 pin_id2node_id.accessor<int64_t,
//                                                                                 1>(), pin_pos.accessor<scalar_t,
//                                                                                 2>(), num_pins,
//                                                                                 at::get_num_threads());
//                                }));

//     float inv_gamma = 1 / gamma;
//     AT_DISPATCH_FLOATING_TYPES(pin_pos.scalar_type(), "wa_wirelength_hpwl", ([&] {
//                                    wa_wirelength_hpwl_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                                        hyperedge_list.accessor<int64_t, 1>(),
//                                                                        hyperedge_list_end.accessor<int64_t, 1>(),
//                                                                        net_mask.accessor<bool, 1>(),
//                                                                        partial_wa_wl.accessor<scalar_t, 2>(),
//                                                                        partial_hpwl.accessor<scalar_t, 2>(),
//                                                                        pin_grad.accessor<scalar_t, 2>(),
//                                                                        num_nets,
//                                                                        inv_gamma,
//                                                                        at::get_num_threads());
//                                }));
//     auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()));
//     const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 2});
//     node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);

//     return make_tuple(partial_wa_wl, node_grad, partial_hpwl);
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor masked_scale_hpwl(torch::Tensor node_pos,
//                                 torch::Tensor pin_id2node_id,
//                                 torch::Tensor pin_rel_cpos,
//                                 torch::Tensor hyperedge_list,
//                                 torch::Tensor hyperedge_list_end,
//                                 torch::Tensor net_mask,
//                                 torch::Tensor hpwl_scale) {
//     CHECK_INPUT(node_pos);

//     const auto num_nodes = node_pos.size(0);
//     const auto num_pins = pin_id2node_id.size(0);
//     const auto num_nets = hyperedge_list_end.size(0);
//     const auto num_channels = 2;  // x, y

//     auto pin_pos = pin_rel_cpos.clone();  // pin
//     auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_forward", ([&] {
//                                    node_pos_to_pin_pos_forward_kernel<scalar_t>(node_pos.accessor<scalar_t, 2>(),
//                                                                                 pin_id2node_id.accessor<int64_t,
//                                                                                 1>(), pin_pos.accessor<scalar_t,
//                                                                                 2>(), num_pins,
//                                                                                 at::get_num_threads());
//                                }));

//     AT_DISPATCH_FLOATING_TYPES(pin_pos.scalar_type(), "wa_wirelength_hpwl", ([&] {
//                                    masked_scale_hpwl_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                                       hyperedge_list.accessor<int64_t, 1>(),
//                                                                       hyperedge_list_end.accessor<int64_t, 1>(),
//                                                                       net_mask.accessor<bool, 1>(),
//                                                                       hpwl_scale.accessor<scalar_t, 1>(),
//                                                                       partial_hpwl.accessor<scalar_t, 2>(),
//                                                                       num_nets,
//                                                                       at::get_num_threads());
//                                }));

//     const auto total_hpwl = partial_hpwl.sum();
//     return total_hpwl;
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void hpwl_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
//                  const torch::TensorAccessor<int64_t, 1> hyperedge_list,
//                  const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
//                  torch::TensorAccessor<scalar_t, 2> partial_hpwl,
//                  int num_nets,
//                  const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nets; ++i) {
//         for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//             int64_t start_idx = 0;
//             if (i != 0) {
//                 start_idx = hyperedge_list_end[i - 1];
//             }
//             int64_t end_idx = hyperedge_list_end[i];
//             partial_hpwl[i][c] = 0;
//             if (end_idx != start_idx) {
//                 int64_t pin_id = hyperedge_list[start_idx];
//                 scalar_t x_min = pin_pos[pin_id][c];
//                 scalar_t x_max = pin_pos[pin_id][c];
//                 for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     x_min = min(xx, x_min);
//                     x_max = max(xx, x_max);
//                 }
//                 partial_hpwl[i][c] = abs(x_max - x_min);
//             }
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor get_hpwl(PlaceData& data, torch::Tensor pin_pos) {
//     CHECK_INPUT(pin_pos);

//     torch::Tensor hyperedge_list = data.hyperedge_list;
//     torch::Tensor hyperedge_list_end = data.hyperedge_list_end;

//     const auto num_nets = hyperedge_list_end.size(0);
//     const int num_channels = 2;
//     auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
//                               hpwl_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                     hyperedge_list.accessor<int64_t, 1>(),
//                                                     hyperedge_list_end.accessor<int64_t, 1>(),
//                                                     partial_hpwl.accessor<scalar_t, 2>(),
//                                                     num_nets,
//                                                     at::get_num_threads());
//                           }));

//     return torch::round(partial_hpwl * (data.__die_scale__ / data.site_width)).sum(1).unsqueeze(1);
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor nodePosToPinPos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos) {
//     CHECK_INPUT(node_pos);

//     const auto num_pins = pin_id2node_id.size(0);
//     auto pin_pos = pin_rel_cpos.clone();  // pin

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_forward", ([&] {
//                                    node_pos_to_pin_pos_forward_kernel<scalar_t>(node_pos.accessor<scalar_t, 2>(),
//                                                                                 pin_id2node_id.accessor<int64_t,
//                                                                                 1>(), pin_pos.accessor<scalar_t,
//                                                                                 2>(), num_pins,
//                                                                                 at::get_num_threads());
//                                }));
//     return pin_pos;
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void node_pos_to_pin_pos_cross_chip_kernel(const torch::TensorAccessor<scalar_t, 2> node_pos,
//                                            const torch::TensorAccessor<int, 1> node_die,
//                                            const torch::TensorAccessor<int64_t, 1> pin_id2node_id,
//                                            torch::TensorAccessor<scalar_t, 2> pin_pos,
//                                            torch::TensorAccessor<int, 1> pin_die,
//                                            int num_pins,
//                                            const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_pins; ++i) {
//         int64_t node_id = pin_id2node_id[i];

//         for (int c = 0; c < 2; ++c) {               // FIXME: channel index
//             pin_pos[i][c] += node_pos[node_id][c];  // TODO: atomic_add
//             pin_die[i] = node_die[node_id];
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void hpwl_cross_chip_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
//                             const torch::TensorAccessor<int, 1> pin_die,
//                             const torch::TensorAccessor<int64_t, 1> hyperedge_list,
//                             const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
//                             torch::TensorAccessor<scalar_t, 3> partial_hpwl,
//                             int num_nets,
//                             const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nets; ++i) {
//         for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//             int64_t start_idx = 0;
//             if (i != 0) {
//                 start_idx = hyperedge_list_end[i - 1];
//             }
//             int64_t end_idx = hyperedge_list_end[i];
//             partial_hpwl[0][i][c] = 0;
//             partial_hpwl[1][i][c] = 0;

//             int64_t bot_count = 0;
//             int64_t top_count = 0;
//             scalar_t x_min_bot = 0;
//             scalar_t x_max_bot = 0;
//             scalar_t x_min_top = 0;
//             scalar_t x_max_top = 0;

//             for (int64_t idx = start_idx; idx < end_idx; idx++) {
//                 int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
//                 if (pin_die[pin_id] == 0 || pin_die[pin_id] == 2) {
//                     if (bot_count == 0) {
//                         x_min_bot = pin_pos[pin_id][c];
//                         x_max_bot = pin_pos[pin_id][c];
//                     } else {
//                         scalar_t xx = pin_pos[pin_id][c];
//                         x_min_bot = min(xx, x_min_bot);
//                         x_max_bot = max(xx, x_max_bot);
//                     }
//                     bot_count++;
//                 }
//                 if (pin_die[pin_id] == 1 || pin_die[pin_id] == 2) {  // FIXME: fix bug: if <- else if
//                     if (top_count == 0) {
//                         x_min_top = pin_pos[pin_id][c];
//                         x_max_top = pin_pos[pin_id][c];
//                     } else {
//                         scalar_t xx = pin_pos[pin_id][c];
//                         x_min_top = min(xx, x_min_top);
//                         x_max_top = max(xx, x_max_top);
//                     }
//                     top_count++;
//                 }
//             }
//             partial_hpwl[0][i][c] = abs(x_max_bot - x_min_bot);
//             partial_hpwl[1][i][c] = abs(x_max_top - x_min_top);
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// tuple<torch::Tensor, torch::Tensor> get_hpwl_cross_chip(PlaceData& data,
//                                                         torch::Tensor node_pos,
//                                                         torch::Tensor node_die) {
//     torch::Tensor pin_id2node_id = data.pin_id2node_id;
//     torch::Tensor hyperedge_list = data.hyperedge_list;
//     torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
//     torch::Tensor pin_rel_cpos = data.pin_rel_cpos;  // TODO: exact rel_cpos

//     const auto num_nodes = node_pos.size(0);
//     const auto num_pins = pin_id2node_id.size(0);
//     const auto num_nets = hyperedge_list_end.size(0);
//     const auto num_channels = 2;  // x, y

//     auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
//     auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt));
//     auto partial_hpwl = torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
//                                    node_pos_to_pin_pos_cross_chip_kernel<scalar_t>(
//                                        node_pos.accessor<scalar_t, 2>(),
//                                        node_die.accessor<int, 1>(),
//                                        pin_id2node_id.accessor<int64_t, 1>(),
//                                        pin_pos.accessor<scalar_t, 2>(),
//                                        pin_die.accessor<int, 1>(),
//                                        num_pins,
//                                        at::get_num_threads());
//                                }));

//     AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
//                               hpwl_cross_chip_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                                pin_die.accessor<int, 1>(),
//                                                                hyperedge_list.accessor<int64_t, 1>(),
//                                                                hyperedge_list_end.accessor<int64_t, 1>(),
//                                                                partial_hpwl.accessor<scalar_t, 3>(),
//                                                                num_nets,
//                                                                at::get_num_threads());
//                           }));

//     return {torch::sum((partial_hpwl.index({0, "..."})).sum(1)), torch::sum((partial_hpwl.index({1,
//     "..."})).sum(1))};
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void net_c_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
//                   const torch::TensorAccessor<int64_t, 1> hyperedge_list,
//                   const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
//                   torch::TensorAccessor<scalar_t, 2> net_c,
//                   int num_nets,
//                   const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nets; ++i) {
//         for (int c = 0; c < 2; ++c) {  // FIXME: channel index
//             int64_t start_idx = 0;
//             if (i != 0) {
//                 start_idx = hyperedge_list_end[i - 1];
//             }
//             int64_t end_idx = hyperedge_list_end[i];
//             net_c[i][c] = 0;
//             if (end_idx != start_idx) {
//                 int64_t pin_id = hyperedge_list[start_idx];
//                 scalar_t x_min = pin_pos[pin_id][c];
//                 scalar_t x_max = pin_pos[pin_id][c];
//                 for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
//                     scalar_t xx = pin_pos[hyperedge_list[idx]][c];
//                     x_min = min(xx, x_min);
//                     x_max = max(xx, x_max);
//                 }
//                 net_c[i][c] = (x_max + x_min) / 2;
//             }
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor get_net_center(PlaceData& data, torch::Tensor pin_pos) {
//     CHECK_INPUT(pin_pos);

//     torch::Tensor hyperedge_list = data.hyperedge_list;
//     torch::Tensor hyperedge_list_end = data.hyperedge_list_end;

//     const auto num_nets = hyperedge_list_end.size(0);
//     const int num_channels = 2;
//     auto net_c = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_FLOATING_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
//                                    net_c_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                           hyperedge_list.accessor<int64_t, 1>(),
//                                                           hyperedge_list_end.accessor<int64_t, 1>(),
//                                                           net_c.accessor<scalar_t, 2>(),
//                                                           num_nets,
//                                                           at::get_num_threads());
//                                }));

//     return net_c;
// }  // END MODULE

// //---------------------------------------------------------------------

// tuple<torch::Tensor, torch::Tensor> get_hpwl_formatted(PlaceData& data,
//                                                        torch::Tensor node_pos,
//                                                        torch::Tensor node_die) {
//     torch::Tensor pin_id2node_id = data.pin_id2node_id;
//     torch::Tensor hyperedge_list = data.hyperedge_list;
//     torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
//     torch::Tensor pin_rel_cpos = data.pin_rel_cpos;  // TODO: exact rel_cpos

//     const auto num_nodes = node_pos.size(0);
//     const auto num_pins = pin_id2node_id.size(0);
//     const auto num_nets = hyperedge_list_end.size(0);
//     const auto num_channels = 2;  // x, y

//     auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
//     auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt));

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
//                                    node_pos_to_pin_pos_cross_chip_kernel<scalar_t>(
//                                        node_pos.accessor<scalar_t, 2>(),
//                                        node_die.accessor<int, 1>(),
//                                        pin_id2node_id.accessor<int64_t, 1>(),
//                                        pin_pos.accessor<scalar_t, 2>(),
//                                        pin_die.accessor<int, 1>(),
//                                        num_pins,
//                                        at::get_num_threads());
//                                }));
//     pin_pos = torch::_cast_Int(pin_pos.clone());
//     auto partial_hpwl = torch::zeros({2, num_nets, num_channels}, torch::dtype(pin_pos.dtype()));

//     AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
//                               hpwl_cross_chip_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
//                                                                pin_die.accessor<int, 1>(),
//                                                                hyperedge_list.accessor<int64_t, 1>(),
//                                                                hyperedge_list_end.accessor<int64_t, 1>(),
//                                                                partial_hpwl.accessor<scalar_t, 3>(),
//                                                                num_nets,
//                                                                at::get_num_threads());
//                           }));
//     // TODO: hardcode die_scale & site_width
//     return {torch::sum((partial_hpwl.index({0, "..."})).sum(1)), torch::sum((partial_hpwl.index({1,
//     "..."})).sum(1))};
// }  // END MODULE

// }  // namespace wa_wirelength_hpwl