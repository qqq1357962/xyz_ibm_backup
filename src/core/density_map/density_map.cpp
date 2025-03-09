
// #include "density_map.h"

// template <typename scalar_t>
// scalar_t overlap(scalar_t x_l, scalar_t x_h, scalar_t bin_x_l) {
//     // bin_x_h == bin_x_l + 1
//     return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
// }

// template <typename scalar_t, typename AtomicOp>
// void density_map_forward_kernel(const torch::TensorAccessor<scalar_t, 2> normalize_node_info,
//                                 const torch::TensorAccessor<int64_t, 1> sorted_node_map,
//                                 torch::TensorAccessor<scalar_t, 2> aux_mat,
//                                 int num_nodes,
//                                 int num_bin_x,
//                                 int num_bin_y,
//                                 const int num_threads,
//                                 AtomicOp atomic_add_op) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int index = 0; index < num_nodes; ++index) {
//         const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
//         const scalar_t weight = normalize_node_info[i][4];
//         if (weight > 0) {
//             const scalar_t x_l = normalize_node_info[i][0];
//             const scalar_t x_h = normalize_node_info[i][1];
//             const scalar_t y_l = normalize_node_info[i][2];
//             const scalar_t y_h = normalize_node_info[i][3];
//             int x_lf = lround(floor(x_l));
//             int x_hf = lround(floor(x_h));
//             int y_lf = lround(floor(y_l));
//             int y_hf = lround(floor(y_h));
//             x_lf = max(x_lf, 0);
//             x_hf = min(x_hf, num_bin_x - 1);
//             y_lf = max(y_lf, 0);
//             y_hf = min(y_hf, num_bin_y - 1);

//             for (int j = x_lf; j < x_hf + 1; j++) {
//                 scalar_t bin_x_l = static_cast<scalar_t>(j);
//                 scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
//                 for (int k = y_lf; k < y_hf + 1; k++) {
//                     scalar_t bin_y_l = static_cast<scalar_t>(k);
//                     scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
//                     scalar_t overlap_area = overlap_x * overlap_y;

//                     atomic_add_op(&aux_mat[j][k], weight * overlap_area);
//                 }
//             }
//         }
//     }
// }  // END MODULE

// template <typename scalar_t, typename AtomicOp>
// void density_map_backward_kernel(const torch::TensorAccessor<scalar_t, 2> normalize_node_info,
//                                  const torch::TensorAccessor<scalar_t, 3> grad_mat,
//                                  const torch::TensorAccessor<int64_t, 1> sorted_node_map,
//                                  torch::TensorAccessor<scalar_t, 2> node_grad,
//                                  float grad_weight,
//                                  int num_bin_x,
//                                  int num_bin_y,
//                                  int num_nodes,
//                                  const int num_threads,
//                                  AtomicOp atomic_add_op) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int index = 0; index < num_nodes; ++index) {
//         const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
//         const scalar_t weight = normalize_node_info[i][4];
//         if (weight > 0) {
//             const scalar_t x_l = normalize_node_info[i][0];
//             const scalar_t x_h = normalize_node_info[i][1];
//             const scalar_t y_l = normalize_node_info[i][2];
//             const scalar_t y_h = normalize_node_info[i][3];
//             int x_lf = lround(floor(x_l));
//             int x_hf = lround(floor(x_h));
//             int y_lf = lround(floor(y_l));
//             int y_hf = lround(floor(y_h));
//             x_lf = max(x_lf, 0);
//             x_hf = min(x_hf, num_bin_x - 1);
//             y_lf = max(y_lf, 0);
//             y_hf = min(y_hf, num_bin_y - 1);

//             // extern __shared__ unsigned char grad_xy[];
//             // scalar_t *grad_x = (scalar_t *)grad_xy;
//             // scalar_t *grad_y = grad_x + blockDim.z;
//             // if (threadIdx.x == 0 && threadIdx.y == 0) {
//             //     grad_x[threadIdx.z] = grad_y[threadIdx.z] = 0;
//             // }
//             // __syncthreads();
//             scalar_t grad_x = 0;
//             scalar_t grad_y = 0;

//             scalar_t part_grad_x = 0;
//             scalar_t part_grad_y = 0;

//             for (int j = x_lf; j < x_hf + 1; j++) {
//                 scalar_t bin_x_l = static_cast<scalar_t>(j);
//                 scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
//                 for (int k = y_lf; k < y_hf + 1; k++) {
//                     scalar_t bin_y_l = static_cast<scalar_t>(k);
//                     scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
//                     scalar_t overlap_area = overlap_x * overlap_y;
//                     scalar_t tmp_x = grad_mat[0][j][k];
//                     scalar_t tmp_y = grad_mat[1][j][k];
//                     // part_grad_x += overlap_area * grad_mat[0][j][k];
//                     // part_grad_y += overlap_area * grad_mat[1][j][k];
//                     part_grad_x += overlap_area * tmp_x;
//                     part_grad_y += overlap_area * tmp_y;
//                 }
//             }
//             // gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
//             // gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
//             // __syncthreads();
//             atomic_add_op(&grad_x, part_grad_x);
//             atomic_add_op(&grad_y, part_grad_y);

//             // if (threadIdx.x == 0 && threadIdx.y == 0) {
//             //     node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
//             //     node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
//             // }
//             node_grad[i][0] = grad_weight * weight * grad_x;
//             node_grad[i][1] = grad_weight * weight * grad_y;
//         }
//     }
// }

// //---------------------------------------------------------------------

// template <typename scalar_t, typename AtomicOp>
// void density_map_forward_naive_kernel(const at::TensorAccessor<scalar_t, 2> node_pos,
//                                       const at::TensorAccessor<scalar_t, 2> node_size,
//                                       const at::TensorAccessor<scalar_t, 1> node_weight,
//                                       const at::TensorAccessor<scalar_t, 1> unit_len,
//                                       at::TensorAccessor<scalar_t, 2> aux_mat,
//                                       int num_bin_x,
//                                       int num_bin_y,
//                                       int num_nodes,
//                                       float min_node_w,
//                                       float min_node_h,
//                                       float margin,
//                                       bool clamp_node,
//                                       const int num_threads,
//                                       AtomicOp atomic_add_op) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nodes; ++i) {
//         scalar_t node_w = node_size[i][0];
//         scalar_t node_h = node_size[i][1];
//         scalar_t ratio = 1.0;
//         if (clamp_node) {
//             const scalar_t node_area = node_w * node_h;
//             node_w = max(node_w, static_cast<scalar_t>(min_node_w));
//             node_h = max(node_h, static_cast<scalar_t>(min_node_h));
//             ratio = node_area / (node_w * node_h);
//         }
//         const scalar_t mgn = static_cast<scalar_t>(margin);
//         const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
//         const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
//         const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);

//         scalar_t x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
//         scalar_t x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
//         scalar_t y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
//         scalar_t y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
//         x_l = min(x_l, num_bin_x_minus_mgn);
//         x_h = max(x_h, mgn);
//         y_l = min(y_l, num_bin_y_minus_mgn);
//         y_h = max(y_h, mgn);
//         if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
//             continue;
//         }
//         const scalar_t p_node_wght = node_weight[i] * ratio;

//         const int x_lf = lround(floor(x_l));
//         const int x_hf = lround(floor(x_h));
//         const int y_lf = lround(floor(y_l));
//         const int y_hf = lround(floor(y_h));

//         for (int j = x_lf; j < x_hf + 1; j++) {
//             const scalar_t bin_x_l = j;
//             const scalar_t bin_x_h = j + 1;
//             scalar_t overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
//             for (int k = y_lf; k < y_hf + 1; k++) {
//                 const scalar_t bin_y_l = k;
//                 const scalar_t bin_y_h = k + 1;
//                 scalar_t overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
//                 scalar_t overlap_area = overlap_x * overlap_y;

//                 atomic_add_op(&aux_mat[j][k], p_node_wght * overlap_area);
//             }
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// template <typename scalar_t>
// void density_map_normalize_node_kernel(const at::TensorAccessor<scalar_t, 2> node_pos,
//                                        const at::TensorAccessor<scalar_t, 2> node_size,
//                                        const at::TensorAccessor<scalar_t, 1> node_weight,
//                                        const at::TensorAccessor<scalar_t, 1> expand_ratio,
//                                        const at::TensorAccessor<scalar_t, 1> unit_len,
//                                        at::TensorAccessor<scalar_t, 2> normalize_node_info,
//                                        int num_bin_x,
//                                        int num_bin_y,
//                                        int num_nodes,
//                                        const int num_threads) {
// #pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
//     for (int i = 0; i < num_nodes; ++i) {
//         normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
//         normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
//         normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
//         normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
//         normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
//         if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
//             normalize_node_info[i][3] - normalize_node_info[i][2] < 0) {
//             normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
//         }
//     }
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor density_map_forward(torch::Tensor normalize_node_info,
//                                   torch::Tensor sorted_node_map,
//                                   torch::Tensor aux_mat,
//                                   int num_bin_x,
//                                   int num_bin_y,
//                                   int num_nodes) {
//     CHECK_INPUT(normalize_node_info);

//     AT_DISPATCH_FLOATING_TYPES(normalize_node_info.scalar_type(), "density_map_forward", ([&] {
//                                    AtomicAdd<scalar_t> atomic_add_op;
//                                    density_map_forward_kernel<scalar_t, decltype(atomic_add_op)>(
//                                        normalize_node_info.accessor<scalar_t, 2>(),
//                                        sorted_node_map.accessor<int64_t, 1>(),
//                                        aux_mat.accessor<scalar_t, 2>(),
//                                        num_nodes,
//                                        num_bin_x,
//                                        num_bin_y,
//                                        at::get_num_threads(),
//                                        atomic_add_op);
//                                }));
//     return aux_mat;
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor density_map_backward(torch::Tensor normalize_node_info,
//                                    torch::Tensor grad_mat,
//                                    torch::Tensor sorted_node_map,
//                                    torch::Tensor node_grad,
//                                    float grad_weight,
//                                    int num_bin_x,
//                                    int num_bin_y,
//                                    int num_nodes) {
//     CHECK_INPUT(normalize_node_info);

//     AT_DISPATCH_FLOATING_TYPES(normalize_node_info.scalar_type(), "density_map_forward", ([&] {
//                                    AtomicAdd<scalar_t> atomic_add_op;
//                                    density_map_backward_kernel<scalar_t, decltype(atomic_add_op)>(
//                                        normalize_node_info.accessor<scalar_t, 2>(),
//                                        grad_mat.accessor<scalar_t, 3>(),
//                                        sorted_node_map.accessor<int64_t, 1>(),
//                                        node_grad.accessor<scalar_t, 2>(),
//                                        grad_weight,
//                                        num_bin_x,
//                                        num_bin_y,
//                                        num_nodes,
//                                        at::get_num_threads(),
//                                        atomic_add_op);
//                                }));
//     return node_grad;
// }  // END MODULE

// //---------------------------------------------------------------------

// at::Tensor density_map_forward_naive(at::Tensor node_pos,
//                                      at::Tensor node_size,
//                                      at::Tensor node_weight,
//                                      at::Tensor unit_len,
//                                      at::Tensor aux_mat,
//                                      int num_bin_x,
//                                      int num_bin_y,
//                                      int num_nodes,
//                                      float min_node_w,
//                                      float min_node_h,
//                                      float margin,
//                                      bool clamp_node) {
//     CHECK_INPUT(node_pos);

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "density_map_forward_naive", ([&] {
//                                    AtomicAdd<scalar_t> atomic_add_op;
//                                    density_map_forward_naive_kernel<scalar_t, decltype(atomic_add_op)>(
//                                        node_pos.accessor<scalar_t, 2>(),
//                                        node_size.accessor<scalar_t, 2>(),
//                                        node_weight.accessor<scalar_t, 1>(),
//                                        unit_len.accessor<scalar_t, 1>(),
//                                        aux_mat.accessor<scalar_t, 2>(),
//                                        num_bin_x,
//                                        num_bin_y,
//                                        num_nodes,
//                                        min_node_w,
//                                        min_node_h,
//                                        margin,
//                                        clamp_node,
//                                        at::get_num_threads(),
//                                        atomic_add_op);
//                                }));
//     return aux_mat;
// }  // END MODULE

// //---------------------------------------------------------------------

// torch::Tensor density_map_normalize_node(torch::Tensor node_pos,
//                                          torch::Tensor node_size,
//                                          torch::Tensor node_weight,
//                                          torch::Tensor expand_ratio,
//                                          torch::Tensor unit_len,
//                                          torch::Tensor normalize_node_info,
//                                          int num_bin_x,
//                                          int num_bin_y,
//                                          int num_nodes) {
//     CHECK_INPUT(node_pos);

//     AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "density_map_normalize_node", ([&] {
//                                    density_map_normalize_node_kernel<scalar_t>(
//                                        node_pos.accessor<scalar_t, 2>(),
//                                        node_size.accessor<scalar_t, 2>(),
//                                        node_weight.accessor<scalar_t, 1>(),
//                                        expand_ratio.accessor<scalar_t, 1>(),
//                                        unit_len.accessor<scalar_t, 1>(),
//                                        normalize_node_info.accessor<scalar_t, 2>(),
//                                        num_bin_x,
//                                        num_bin_y,
//                                        num_nodes,
//                                        at::get_num_threads());
//                                }));

//     return normalize_node_info;
// }  // END MODULE