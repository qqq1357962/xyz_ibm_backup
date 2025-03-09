#include "rudy_map.h"

template <typename scalar_t, typename AtomicOp>
void rudy_map_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
                     const torch::TensorAccessor<int64_t, 1> hyperedge_list,
                     const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
                     const torch::TensorAccessor<scalar_t, 1> unit_len,
                     torch::TensorAccessor<scalar_t, 2> horizontal_map,
                     torch::TensorAccessor<scalar_t, 2> vertical_map,
                     int num_bin_x,
                     int num_bin_y,
                     int num_nets,
                     float margin,
                     const int num_threads,
                     AtomicOp atomic_add_op) {
    // margin
    const scalar_t mgn = static_cast<scalar_t>(margin);
    const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
    const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
    const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);
#pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];

        if (end_idx != start_idx) {
            scalar_t weight = end_idx - start_idx;
            weight = 1;
            int64_t pin_id = hyperedge_list[start_idx];
            scalar_t x_min = pin_pos[pin_id][0];
            scalar_t x_max = pin_pos[pin_id][0];
            scalar_t y_min = pin_pos[pin_id][1];
            scalar_t y_max = pin_pos[pin_id][1];

            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                scalar_t xx = pin_pos[hyperedge_list[idx]][0];
                scalar_t yy = pin_pos[hyperedge_list[idx]][1];
                x_min = min(xx, x_min);
                x_max = max(xx, x_max);
                y_min = min(yy, y_min);
                y_max = max(yy, y_max);
            }

            // bin_index
            scalar_t x_l = max(x_min / unit_len[0], mgn);
            scalar_t x_h = min(x_max / unit_len[0], num_bin_x_minus_mgn);
            scalar_t y_l = max(y_min / unit_len[1], mgn);
            scalar_t y_h = min(y_max / unit_len[1], num_bin_y_minus_mgn);
            x_l = min(x_l, num_bin_x_minus_mgn);
            x_h = max(x_h, mgn);
            y_l = min(y_l, num_bin_y_minus_mgn);
            y_h = max(y_h, mgn);

            const int x_lf = lround(floor(x_l));
            const int x_hf = lround(floor(x_h));
            const int y_lf = lround(floor(y_l));
            const int y_hf = lround(floor(y_h));

            for (int j = x_lf; j < x_hf + 1; j++) {
                const scalar_t bin_x_l = j;
                const scalar_t bin_x_h = j + 1;
                scalar_t overlap_x = (min(x_h, bin_x_h) - max(x_l, bin_x_l));
                for (int k = y_lf; k < y_hf + 1; k++) {
                    const scalar_t bin_y_l = k;
                    const scalar_t bin_y_h = k + 1;
                    scalar_t overlap_y = (min(y_h, bin_y_h) - max(y_l, bin_y_l));

                    scalar_t overlap = overlap_x * overlap_y * weight;
                    // scalar_t overlap_area = overlap_x * overlap_y;
                    if (overlap_y != 0)
                        atomic_add_op(&horizontal_map[j][k], overlap / (y_h - y_l));
                    else
                        atomic_add_op(&horizontal_map[j][k], overlap_x * weight);
                    if (overlap_x != 0)
                        atomic_add_op(&vertical_map[j][k], overlap / (x_h - x_l));
                    else
                        atomic_add_op(&horizontal_map[j][k], overlap_y * weight);
                }
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void rudy_map_forward_naive(PlaceData& data,
                            torch::Tensor pin_pos,
                            torch::Tensor unit_len,
                            torch::Tensor& horizontal_map,
                            torch::Tensor& vertical_map,
                            int num_bin_x,
                            int num_bin_y,
                            float margin) {
    CHECK_INPUT(pin_pos);

    torch::Tensor hyperedge_list = data.hyperedge_list;
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
    const auto num_nets = hyperedge_list_end.size(0);

    AT_DISPATCH_FLOATING_TYPES(pin_pos.scalar_type(), "rudy_map_forward_naive", ([&] {
                                   AtomicAdd<scalar_t> atomic_add_op;
                                   rudy_map_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
                                                             hyperedge_list.accessor<int64_t, 1>(),
                                                             hyperedge_list_end.accessor<int64_t, 1>(),
                                                             unit_len.accessor<scalar_t, 1>(),
                                                             horizontal_map.accessor<scalar_t, 2>(),
                                                             vertical_map.accessor<scalar_t, 2>(),
                                                             num_bin_x,
                                                             num_bin_y,
                                                             num_nets,
                                                             margin,
                                                             at::get_num_threads(),
                                                             atomic_add_op);
                               }));
}

template <typename scalar_t, typename AtomicOp>
void pin_density_map_kernel(const torch::TensorAccessor<scalar_t, 2> pin_pos,
                            const torch::TensorAccessor<int64_t, 1> hyperedge_list,
                            const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
                            const torch::TensorAccessor<scalar_t, 1> unit_len,
                            torch::TensorAccessor<scalar_t, 2> aux_mat,
                            int num_bin_x,
                            int num_bin_y,
                            int num_nets,
                            float margin,
                            const int num_threads,
                            AtomicOp atomic_add_op) {
    // margin
    const scalar_t mgn = static_cast<scalar_t>(margin);
    const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
    const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
    const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);
#pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];

        if (end_idx != start_idx) {
            scalar_t weight = end_idx - start_idx;
            weight = 1;  // TODO:
            int64_t pin_id = hyperedge_list[start_idx];
            scalar_t x_min = pin_pos[pin_id][0];
            scalar_t x_max = pin_pos[pin_id][0];
            scalar_t y_min = pin_pos[pin_id][1];
            scalar_t y_max = pin_pos[pin_id][1];

            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                scalar_t xx = pin_pos[hyperedge_list[idx]][0];
                scalar_t yy = pin_pos[hyperedge_list[idx]][1];
                x_min = min(xx, x_min);
                x_max = max(xx, x_max);
                y_min = min(yy, y_min);
                y_max = max(yy, y_max);
            }

            // bin_index
            scalar_t x_l = max(x_min / unit_len[0], mgn);
            scalar_t x_h = min(x_max / unit_len[0], num_bin_x_minus_mgn);
            scalar_t y_l = max(y_min / unit_len[1], mgn);
            scalar_t y_h = min(y_max / unit_len[1], num_bin_y_minus_mgn);
            x_l = min(x_l, num_bin_x_minus_mgn);
            x_h = max(x_h, mgn);
            y_l = min(y_l, num_bin_y_minus_mgn);
            y_h = max(y_h, mgn);

            const int x_lf = lround(floor(x_l));
            const int x_hf = lround(floor(x_h));
            const int y_lf = lround(floor(y_l));
            const int y_hf = lround(floor(y_h));

            weight = 1 / (y_h - y_l) * (x_h - x_l);

            for (int j = x_lf; j < x_hf + 1; j++) {
                const scalar_t bin_x_l = j;
                const scalar_t bin_x_h = j + 1;
                scalar_t overlap_x = (min(x_h, bin_x_h) - max(x_l, bin_x_l));
                for (int k = y_lf; k < y_hf + 1; k++) {
                    const scalar_t bin_y_l = k;
                    const scalar_t bin_y_h = k + 1;
                    scalar_t overlap_y = (min(y_h, bin_y_h) - max(y_l, bin_y_l));

                    // scalar_t overlap = overlap_x * overlap_y * weight;
                    // // scalar_t overlap_area = overlap_x * overlap_y;
                    // if (overlap_y != 0)
                    //     atomic_add_op(&horizontal_map[j][k], overlap / (y_h - y_l));
                    // else
                    //     atomic_add_op(&horizontal_map[j][k], overlap_x * weight);
                    // if (overlap_x != 0)
                    //     atomic_add_op(&vertical_map[j][k], overlap / (x_h - x_l));
                    // else
                    //     atomic_add_op(&horizontal_map[j][k], overlap_y * weight);

                    if (overlap_y != 0)
                        atomic_add_op(&aux_mat[j][k], overlap_x / (x_h - x_l));
                    else if (overlap_x != 0)
                        atomic_add_op(&aux_mat[j][k], overlap_y / (y_h - y_l));
                    else
                        atomic_add_op(&aux_mat[j][k], overlap_x * overlap_y * weight);
                }
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void pin_density_map_forward_naive(PlaceData& data,
                                   torch::Tensor pin_pos,
                                   torch::Tensor unit_len,
                                   torch::Tensor& aux_mat,
                                   int num_bin_x,
                                   int num_bin_y,
                                   float margin) {
    torch::Tensor hyperedge_list = data.hyperedge_list.to(torch::kCPU);
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end.to(torch::kCPU);
    const auto num_nets = hyperedge_list_end.size(0);

    AT_DISPATCH_FLOATING_TYPES(pin_pos.scalar_type(), "pin_density_map_forward_naive", ([&] {
                                   AtomicAdd<scalar_t> atomic_add_op;
                                   pin_density_map_kernel<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
                                                                    hyperedge_list.accessor<int64_t, 1>(),
                                                                    hyperedge_list_end.accessor<int64_t, 1>(),
                                                                    unit_len.accessor<scalar_t, 1>(),
                                                                    aux_mat.accessor<scalar_t, 2>(),
                                                                    num_bin_x,
                                                                    num_bin_y,
                                                                    num_nets,
                                                                    margin,
                                                                    at::get_num_threads(),
                                                                    atomic_add_op);
                               }));
}