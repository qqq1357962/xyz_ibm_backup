#include "post_process.h"

// template <typename scalar_t>
struct coordPoints {
    int id;
    int x;
};
struct rankPoints {
    int id;
    float score;
};

// calculcate the increase of HPWL after placing a via[i] to the bounding box of net[j] on 2 face
template <typename scalar_t>
scalar_t node_a_net_b_offset(int a,
                             int b,
                             torch::TensorAccessor<scalar_t, 4> boundary,
                             torch::TensorAccessor<scalar_t, 2> via) {
    scalar_t sum = 0;
    for (int face = 0; face < 2; face++) {
        for (int axis = 0; axis < 2; axis++) {
            if (via[a][axis] < boundary[face][b][axis][1]) {
                sum = sum + (boundary[face][b][axis][1] - via[a][axis]);
            } else if (via[a][axis] > boundary[face][b][axis][0]) {
                sum = sum + (via[a][axis] - boundary[face][b][axis][0]);
            }
        }
    }
    return sum;
}

bool sort_asc(coordPoints a, coordPoints b) {
    if (a.x < b.x) {
        return true;
    }
    return false;
}

bool sort_dec(rankPoints a, rankPoints b) {
    if (a.score > b.score) {
        return true;
    }
    return false;
}

template <typename scalar_t>
void set_data(vector<rankPoints>& rank,
              vector<coordPoints> coordinates[],
              torch::TensorAccessor<int, 1> skip,
              int64_t y_coord[],
              torch::TensorAccessor<scalar_t, 4> boundary,
              torch::TensorAccessor<scalar_t, 2> via,
              torch::Tensor via_pos,
              torch::Tensor numRows,
              torch::Tensor row_height,
              torch::Tensor row_shift,
              int num_nets) {
    torch::Tensor bin_size_y = torch::_cast_Double(row_height.clone());
    int total_score = 0;

    for (int i = 0; i < num_nets; ++i) {
        if (skip[i] == 0) {
            scalar_t score = node_a_net_b_offset(i, i, boundary, via);
            if (score > 0) {
                rankPoints tmpRank;
                tmpRank.id = i;
                tmpRank.score = score;
                rank.push_back(tmpRank);
                total_score = total_score + score;
            }
            coordPoints tmpCoord;
            tmpCoord.id = i;
            tmpCoord.x = via_pos[i][0].item<int>();
            int bin_id_y = torch::floor((via_pos[i][1] - row_shift) / bin_size_y).item<int>();
            coordinates[bin_id_y].push_back(tmpCoord);
            y_coord[bin_id_y] = via_pos[i][1].item<int>();
        }
    }
    for (int i = 0; i < numRows.item<int>(); ++i) {
        std::sort(coordinates[i].begin(), coordinates[i].end(), sort_asc);
    }

    // logger.info("max improvment  %d", total_score);
}

template <typename scalar_t>
void fill_space(int max_loop,
                torch::TensorAccessor<scalar_t, 4> boundary,
                torch::TensorAccessor<scalar_t, 2> via,
                torch::Tensor via_pos,
                torch::TensorAccessor<int, 1> skip,
                torch::Tensor numRows,
                torch::Tensor row_height,
                torch::Tensor row_shift,
                torch::Tensor size,
                torch::Tensor die_info,
                int num_nets,
                int num_threads) {
    int rowCount = numRows.item<int>();
    int64_t y_coord[rowCount] = {0};
    vector<rankPoints> rank;
    vector<coordPoints> coordinates[rowCount];
    int xl = die_info[0].item<int>();
    int xh = die_info[1].item<int>();
    int yl = die_info[2].item<int>();
    int yh = die_info[3].item<int>();

    set_data(rank, coordinates, skip, y_coord, boundary, via, via_pos, numRows, row_height, row_shift, num_nets);
    // logger.info("counter: %d ", rank.size());
    std::sort(rank.begin(), rank.end(), sort_dec);
    int64_t valid_y = -1;
    int valid_y_index = -1;
    float total_score = 0;
    {
        ///*
        for (int i = 0; i < rank.size(); i++) {
            total_score = total_score + rank[i].score;
        }
        // logger.info("maximmum gain:%f", total_score);
        for (int i = 0; i < rowCount; ++i) {
            if (y_coord[i] > 0) {
                if (valid_y_index == -1) {
                    valid_y_index = i;
                    valid_y = y_coord[i];
                } else {
                    valid_y = min(valid_y, y_coord[i]);
                }
            }
            std::sort(coordinates[i].begin(), coordinates[i].end(), sort_asc);
        }
        // logger.info("cp3");

        for (int64_t i = valid_y_index - 1; i >= 0; --i) {
            int64_t tmp = valid_y_index - i;
            tmp = valid_y - tmp * row_height.item<int>();
            if (y_coord[i] == 0) {
                y_coord[i] = tmp;
            }
        }
        for (int64_t i = valid_y_index + 1; i < rowCount; ++i) {
            int64_t tmp = i - valid_y_index;
            tmp = valid_y + tmp * row_height.item<int>();
            if (y_coord[i] == 0) {
                y_coord[i] = tmp;
            }
        }
        // logger.info("cp2");
        vector<int> y_coordV(y_coord, y_coord + rowCount - 1);
        // logger.info("cp");
        float reduce = 0;
        int move = 0;
        int block;

        for (block = 0; block < rank.size(); ++block) {
            int64_t size_x = size[rank[block].id][0].item<int>();
            int64_t offset_x = size_x / 2;
            if (rank[block].score < 20) {
                break;
            }

            int64_t upper_bound_y = min(boundary[0][rank[block].id][1][0], boundary[1][rank[block].id][1][0]);
            int64_t upper_bound_y_index =
                std::upper_bound(y_coordV.begin(), y_coordV.end(), upper_bound_y) - y_coordV.begin() - 1;
            int64_t lower_bound_y = max(boundary[0][rank[block].id][1][1], boundary[1][rank[block].id][1][1]);
            int64_t lower_bound_y_index =
                std::lower_bound(y_coordV.begin(), y_coordV.end(), lower_bound_y) - y_coordV.begin();
            int64_t upper_bound_x = min(boundary[0][rank[block].id][0][0], boundary[1][rank[block].id][0][0]);
            int64_t lower_bound_x = max(boundary[0][rank[block].id][0][1], boundary[1][rank[block].id][0][1]);
            if (lower_bound_x > upper_bound_x || lower_bound_y > upper_bound_y) {
                continue;
            }

            if (block % 100 == 0) {
                // logger.info("fill_space:%d   %d    %f", block, rank.size(), reduce);
                // std::cout << boundary[0][rank[block].id][1][0] << "  "<< boundary[1][rank[block].id][1][0]<<"   " <<
                // upper_bound_y<<"   " << upper_bound_y_index<< "\n"; std::cout << boundary[0][rank[block].id][1][1] <<
                // "  "<< boundary[1][rank[block].id][1][1]<<"   " << lower_bound_y<<"   " << lower_bound_y_index<<
                // "\n"; std::cout << boundary[0][rank[block].id][0][0] << "  "<< boundary[1][rank[block].id][0][0]<<"
                // " << upper_bound_x<<"   " << "\n"; std::cout << boundary[0][rank[block].id][0][1] << "  "<<
                // boundary[1][rank[block].id][0][1]<<"   " << lower_bound_x<<"   " << "\n";
            }

            int found = 0;

            coordPoints lower_tmp_point;
            lower_tmp_point.x = lower_bound_x;
            coordPoints upper_tmp_point;
            upper_tmp_point.x = upper_bound_x;

            // std::vector<int>::iterator it;
            auto it = coordinates[0].begin();
            int64_t current_y = 0;
            int current_y_index = 0;
            int64_t gap_needed = size_x * 2;
            int64_t x_pos = 0;
            int x_index_offset;
            int code = 0;

            for (int64_t y_index = lower_bound_y_index; y_index < upper_bound_y_index; y_index++) {
                current_y = y_coord[y_index];
                current_y_index = y_index;

                int64_t lower_bound_x_index =
                    std::lower_bound(
                        coordinates[y_index].begin(), coordinates[y_index].end(), lower_tmp_point, sort_asc) -
                    coordinates[y_index].begin();
                int64_t upper_bound_x_index =
                    std::upper_bound(
                        coordinates[y_index].begin(), coordinates[y_index].end(), upper_tmp_point, sort_asc) -
                    coordinates[y_index].begin() - 1;
                if (coordinates[y_index].size() > 0) {
                    // logger.info("low:%d     %d     %d", lower_bound_x_index,
                    // coordinates[y_index][lower_bound_x_index].x, lower_bound_x); logger.info("upp:%d     %d     %d",
                    // upper_bound_x_index, coordinates[y_index][upper_bound_x_index].x, upper_bound_x);
                }

                if (coordinates[y_index].size() == 0) {
                    x_pos = lower_bound_x;
                    x_index_offset = -1;
                    code = 1;
                    if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                        found = 1;
                        break;
                    }
                } else if (lower_bound_x > coordinates[y_index][coordinates[y_index].size() - 1].x) {
                    if (upper_bound_x - lower_bound_x >= gap_needed) {
                        x_pos = max(lower_bound_x + size_x,
                                    coordinates[y_index][coordinates[y_index].size() - 1].x + size_x);
                        x_index_offset = -1;
                        found = 1;
                        code = 2;
                        if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                            found = 1;
                            break;
                        }
                    }

                } else if (upper_bound_x < coordinates[y_index][0].x) {
                    if (upper_bound_x - lower_bound_x >= gap_needed) {
                        x_pos = min(upper_bound_x - size_x, coordinates[y_index][0].x - size_x);
                        x_index_offset = 0;
                        code = 3;
                        if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                            found = 1;
                            break;
                        }
                    }
                } else if (lower_bound_x_index > upper_bound_x_index) {
                    if (coordinates[y_index][upper_bound_x_index].x - coordinates[y_index][lower_bound_x_index].x &&
                        lower_bound_x_index - upper_bound_x_index == 1) {
                        x_pos = max(lower_bound_x, coordinates[y_index][upper_bound_x_index].x + size_x);
                        x_index_offset = lower_bound_x_index;
                        code = 4;
                        if (x_pos <= coordinates[y_index][lower_bound_x_index].x - size_x) {
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    }
                } else {
                    int x_index = 0;

                    if (lower_bound_x_index > 0) {
                        if (coordinates[y_index][lower_bound_x_index].x -
                                coordinates[y_index][lower_bound_x_index - 1].x >=
                            gap_needed) {
                            x_pos = coordinates[y_index][lower_bound_x_index].x - size_x;
                            x_index_offset = lower_bound_x_index;
                            code = 5;
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    } else {
                        if (coordinates[y_index][lower_bound_x_index].x - lower_bound_x >= size_x) {
                            x_pos = coordinates[y_index][lower_bound_x_index].x - size_x;
                            x_index_offset = lower_bound_x_index;
                            code = 6;
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    }

                    for (x_index = lower_bound_x_index + 1; x_index <= upper_bound_x_index; x_index++) {
                        if (coordinates[y_index][x_index].x - coordinates[y_index][x_index - 1].x >= gap_needed) {
                            x_pos = coordinates[y_index][x_index - 1].x + size_x;
                            x_index_offset = x_index;
                            code = 7;
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    }
                    if (upper_bound_x_index < coordinates[y_index].size() - 1 && found == 0) {
                        if (coordinates[y_index][upper_bound_x_index + 1].x -
                                coordinates[y_index][upper_bound_x_index].x >=
                            gap_needed) {
                            x_pos = coordinates[y_index][upper_bound_x_index].x + size_x;
                            x_index_offset = upper_bound_x_index + 1;
                            code = 8;
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    } else if (found == 0) {
                        if (upper_bound_x - coordinates[y_index][upper_bound_x_index].x >= size_x) {
                            x_pos = coordinates[y_index][upper_bound_x_index].x + size_x;
                            x_index_offset = -1;
                            // logger.info("???: %d    %d    %d", coordinates[y_index][upper_bound_x_index].x,
                            // upper_bound_x, lower_bound_x);
                            code = 9;
                            if (x_pos <= xh - offset_x && x_pos >= offset_x + xl) {
                                found = 1;
                                break;
                            }
                        }
                    }
                    if (found == 1) {
                        break;
                    }
                }
                if (found == 1) {
                    break;
                }
            }
            if (found == 1) {
                scalar_t score_original = rank[block].score;
                via_pos[rank[block].id][1] = current_y;
                // std::cout<<rank[block].score <<"\n";
                reduce = reduce + rank[block].score;
                move = move + 1;
                via_pos[rank[block].id][0] = x_pos;
                coordPoints insertPoint;
                insertPoint.id = rank[block].id;
                insertPoint.x = x_pos;
                it = coordinates[current_y_index].begin();
                // logger.info("type %d    %d      %f  %f ", code, rank[block].id, node_a_net_b_offset(rank[block].id,
                // rank[block].id, boundary, via_pos.accessor<scalar_t, 2>()), score_original); logger.info("y: %d  %d
                // %d x: %d  %d %d", via_pos[rank[block].id][1].item<int>(), upper_bound_y, lower_bound_y,
                // via_pos[rank[block].id][0].item<int>(), upper_bound_x, lower_bound_x);
                if (x_index_offset == -1) {
                    coordinates[current_y_index].push_back(insertPoint);
                } else {
                    coordinates[current_y_index].insert(it + x_index_offset, insertPoint);
                }
                for (int i = 0; i < coordinates[current_y_index].size() - 1; i++) {
                    if (coordinates[current_y_index][i].x > coordinates[current_y_index][i + 1].x) {
                        // logger.info("wrong %d    %d", code, current_y_index);
                    }
                }
                if (node_a_net_b_offset(rank[block].id, rank[block].id, boundary, via_pos.accessor<scalar_t, 2>()) >
                    0) {
                    // logger.info(
                    //     "type %d    %d      %f  %f ",
                    //     code,
                    //     rank[block].id,
                    //     node_a_net_b_offset(rank[block].id, rank[block].id, boundary, via_pos.accessor<scalar_t,
                    //     2>()), score_original);
                    // logger.info("y: %d  %d %d x: %d  %d %d",
                    //             via_pos[rank[block].id][1].item<int>(),
                    //             upper_bound_y,
                    //             lower_bound_y,
                    //             via_pos[rank[block].id][0].item<int>(),
                    //             upper_bound_x,
                    //             lower_bound_x);
                }

            } else if (upper_bound_y - lower_bound_y > 200 && upper_bound_x - lower_bound_x > 0) {
                // logger.info("not found: %d    %f    %d    %d    %d     %d",
                // block, rank[block].score, upper_bound_y, lower_bound_y, upper_bound_x, lower_bound_x);
            }
        }

        // logger.info("score reduce = %f", reduce);
        // logger.info("move count = %d", move);
        // logger.info("total node = %d", block);
        //*/
    }
}

template <typename scalar_t>
void node_pos_to_pin_pos_cross_chip_kernel(const torch::TensorAccessor<scalar_t, 2> node_pos,
                                           const torch::TensorAccessor<int, 1> node_die,
                                           const torch::TensorAccessor<int64_t, 1> pin_id2node_id,
                                           torch::TensorAccessor<scalar_t, 2> pin_pos,
                                           torch::TensorAccessor<int, 1> pin_die,
                                           int num_pins,
                                           const int num_threads) {
#pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
    for (int i = 0; i < num_pins; ++i) {
        int64_t node_id = pin_id2node_id[i];

        for (int c = 0; c < 2; ++c) {               // FIXME: channel index
            pin_pos[i][c] += node_pos[node_id][c];  // TODO: atomic_add
            pin_die[i] = node_die[node_id];
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

template <typename scalar_t>
void evaluate_boundary(const torch::TensorAccessor<scalar_t, 2> pin_pos,
                       const torch::TensorAccessor<int, 1> pin_die,
                       const torch::TensorAccessor<int64_t, 1> hyperedge_list,
                       const torch::TensorAccessor<int64_t, 1> hyperedge_list_end,
                       torch::TensorAccessor<scalar_t, 4> boundary,
                       torch::TensorAccessor<scalar_t, 2> via,
                       torch::TensorAccessor<int, 1> skip,
                       int num_nets,
                       const int num_threads) {
#pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i) {
        for (int c = 0; c < 2; ++c) {  // FIXME: channel index
            int64_t start_idx = 0;
            if (i != 0) {
                start_idx = hyperedge_list_end[i - 1];
            }
            int64_t end_idx = hyperedge_list_end[i];

            int64_t bot_count = 0;
            int64_t top_count = 0;
            scalar_t x_min_bot = 0;
            scalar_t x_max_bot = 0;
            scalar_t x_min_top = 0;
            scalar_t x_max_top = 0;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list[idx];  // FIXME: fix bug: idx error
                if (pin_die[pin_id] == 0) {
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
                if (pin_die[pin_id] == 1) {  // FIXME: fix bug: if <- else if
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
                if (pin_die[pin_id] == 2) {
                    via[i][c] = pin_pos[pin_id][c];
                    skip[i] = 0;
                }
            }
            if (skip[i] == 0) {
                // std::cout <<"bot     "<< x_max_bot << "   " << x_min_bot<<"\n";
                // std::cout <<"top     "<< x_max_top << "   " << x_min_top<<"\n";
            }

            boundary[0][i][c][0] = x_max_bot;
            boundary[0][i][c][1] = x_min_bot;
            boundary[1][i][c][0] = x_max_top;
            boundary[1][i][c][1] = x_min_top;
        }
    }

}  // END MODULE

//---------------------------------------------------------------------

template <typename scalar_t>
void brute_force_swap_via(int max_loop,
                          torch::TensorAccessor<scalar_t, 4> boundary,
                          torch::TensorAccessor<scalar_t, 2> via,
                          torch::Tensor via_pos,
                          torch::TensorAccessor<int, 1> skip,
                          int num_nets,
                          int num_threads) {
    float score[num_nets];
    float max_score = node_a_net_b_offset(0, 0, boundary, via);
    for (int i = 0; i < num_nets; i++) {
        score[i] = node_a_net_b_offset(i, i, boundary, via);
        max_score = max(score[i], max_score);
    }
    for (int i = 0; i < num_nets; i++) {
        if (skip[i] == 1) {
            continue;
        }
        int tmp_swap_index = -1;
        scalar_t tmp_swap_value = max_score * 4;
        if (i % 2000 == 1) {
            // logger.info("iter %d", i);
        }
        //#pragma omp parallel for num_threads(num_threads)  // schedule(dynamic, chunk_size)
        for (int j = 0; j < num_nets; j++) {
            if (i != j && skip[j] == 0) {
                scalar_t node_i_net_j_offset = node_a_net_b_offset(i, j, boundary, via);
                scalar_t node_j_net_i_offset = node_a_net_b_offset(j, i, boundary, via);
                scalar_t old_score = score[i] + score[j];
                scalar_t new_score = node_i_net_j_offset + node_j_net_i_offset;

                if (old_score > new_score && tmp_swap_value > new_score) {
                    tmp_swap_value = new_score;
                    tmp_swap_index = j;
                }
            }
        }
        if (tmp_swap_index != -1) {
            scalar_t tmp = via[i][0];
            via[i][0] = via[tmp_swap_index][0];
            via[tmp_swap_index][0] = tmp;
            tmp = via[i][1];
            via[i][1] = via[tmp_swap_index][1];
            via[tmp_swap_index][1] = tmp;
            torch::Tensor tmpTensor = via_pos[i].clone();
            via_pos[i].data().copy_(via_pos[tmp_swap_index].data());
            via_pos[tmp_swap_index].data().copy_(tmpTensor.data());
        }
    }

}  // END MODULE

template <typename scalar_t>
void swap_via_overlap_region(int max_loop,
                             torch::TensorAccessor<scalar_t, 4> boundary,
                             torch::TensorAccessor<scalar_t, 2> via,
                             torch::Tensor via_pos,
                             torch::TensorAccessor<int, 1> skip,
                             torch::Tensor numRows,
                             torch::Tensor row_height,
                             torch::Tensor row_shift,
                             torch::Tensor size,
                             int num_nets,
                             int num_threads) {
    int rowCount = numRows.item<int>();
    int64_t y_coord[rowCount] = {0};
    vector<rankPoints> rank;
    vector<coordPoints> coordinates[rowCount];
    set_data(rank, coordinates, skip, y_coord, boundary, via, via_pos, numRows, row_height, row_shift, num_nets);
    // logger.info("counter: %d ", rank.size());
    std::sort(rank.begin(), rank.end(), sort_dec);

    float reduce = 0;

    int move = 0;
    int block;

    //int64_t coord_map[num_nets][2];
    std::vector<std::vector<int64_t>>coord_map;
    coord_map.resize(num_nets);

    for (int y = 0; y < numRows.item<int>(); y++) {
        for (int x = 0; x < coordinates[y].size(); x++) {
            //coord_map[coordinates[y][x].id][0] = x;
            //coord_map[coordinates[y][x].id][1] = y;
            coord_map[coordinates[y][x].id].push_back(x);
            coord_map[coordinates[y][x].id].push_back(y);
        }
    }
    vector<int> y_coordV(y_coord, y_coord + rowCount - 1);
    int overlap = 0;
    if (rank.size() > 30000) {
        overlap = 1;
        // logger.info("true");

    } else {
        overlap = 0;
        // logger.info("false");
    }

    for (block = 0; block < rank.size(); block++) {
        if (rank[block].score < 20) {
            break;
        }
        int64_t upper_bound_y, lower_bound_y, upper_bound_x, lower_bound_x;
        if (overlap == 1) {
            upper_bound_y = min(boundary[0][rank[block].id][1][0], boundary[1][rank[block].id][1][0]);
            lower_bound_y = max(boundary[0][rank[block].id][1][1], boundary[1][rank[block].id][1][1]);
            upper_bound_x = min(boundary[0][rank[block].id][0][0], boundary[1][rank[block].id][0][0]);
            lower_bound_x = max(boundary[0][rank[block].id][0][1], boundary[1][rank[block].id][0][1]);
        } else {
            upper_bound_y = max(boundary[0][rank[block].id][1][0], boundary[1][rank[block].id][1][0]);
            lower_bound_y = min(boundary[0][rank[block].id][1][1], boundary[1][rank[block].id][1][1]);
            upper_bound_x = max(boundary[0][rank[block].id][0][0], boundary[1][rank[block].id][0][0]);
            lower_bound_x = min(boundary[0][rank[block].id][0][1], boundary[1][rank[block].id][0][1]);
        }
        int64_t upper_bound_y_index =
            std::upper_bound(y_coordV.begin(), y_coordV.end(), upper_bound_y) - y_coordV.begin() - 1;
        int64_t lower_bound_y_index =
            std::lower_bound(y_coordV.begin(), y_coordV.end(), lower_bound_y) - y_coordV.begin();

        if (block % 100 == 0) {
            // logger.info("swap_via:%d   %d    %f", block, rank.size(), reduce);
        }

        int found = -1;
        float score_gain = 0;

        coordPoints lower_tmp_point;
        lower_tmp_point.x = lower_bound_x;
        coordPoints upper_tmp_point;
        upper_tmp_point.x = upper_bound_x;

        // std::vector<int>::iterator it;
        scalar_t min_score = rank[0].score * 2;

        for (int y_index = lower_bound_y_index; y_index < upper_bound_y_index; y_index++) {
            if (coordinates[y_index].size() == 0) {
                break;
            }

            int64_t lower_bound_x_index =
                std::lower_bound(coordinates[y_index].begin(), coordinates[y_index].end(), lower_tmp_point, sort_asc) -
                coordinates[y_index].begin();
            int64_t upper_bound_x_index =
                std::upper_bound(coordinates[y_index].begin(), coordinates[y_index].end(), upper_tmp_point, sort_asc) -
                coordinates[y_index].begin() - 1;

            if (lower_bound_x_index <= upper_bound_x_index) {
                for (int x_index = lower_bound_x_index; x_index <= upper_bound_x_index; x_index++) {
                    scalar_t score_original = rank[block].score + node_a_net_b_offset(coordinates[y_index][x_index].id,
                                                                                      coordinates[y_index][x_index].id,
                                                                                      boundary,
                                                                                      via);
                    scalar_t score_new =
                        node_a_net_b_offset(rank[block].id, coordinates[y_index][x_index].id, boundary, via) +
                        node_a_net_b_offset(coordinates[y_index][x_index].id, rank[block].id, boundary, via);
                    // logger.info("hello: %d %d %d",  min_score, score_original ,score_new);
                    if (score_new < score_original && min_score > score_new) {
                        min_score = score_new;
                        score_gain = score_original - score_new;
                        found = coordinates[y_index][x_index].id;
                    }
                }
            }
        }
        if (found > -1) {
            // std::cout<<"swap"<< rank[block].id + 1<< "  "<< found + 1<< "  "<<"\n";

            // std::cout<<"before"<< via[rank[block].id][0] << "  "<< via[rank[block].id][1] << "  "<< via[found][0]<< "
            // "<< via[found][1]<<"\n";
            scalar_t tmp = via[rank[block].id][0];
            via[rank[block].id][0] = via[found][0];
            via[found][0] = tmp;

            tmp = via[rank[block].id][1];
            via[rank[block].id][1] = via[found][1];
            via[found][1] = tmp;

            torch::Tensor tmpTensor = via_pos[rank[block].id].clone();
            via_pos[rank[block].id].data().copy_(via_pos[found].data());
            via_pos[found].data().copy_(tmpTensor.data());

            coordinates[coord_map[rank[block].id][1]][coord_map[rank[block].id][0]].id = found;
            coordinates[coord_map[found][1]][coord_map[found][0]].id = rank[block].id;

            int tmp2 = coord_map[found][0];
            coord_map[found][0] = coord_map[rank[block].id][0];
            coord_map[rank[block].id][0] = tmp2;

            tmp2 = coord_map[found][1];
            coord_map[found][1] = coord_map[rank[block].id][1];
            coord_map[rank[block].id][1] = tmp2;

            move++;
            reduce = reduce + score_gain;

            // std::cout<<"after"<< via[rank[block].id][0] << "  "<< via[rank[block].id][1] << "  "<< via[found][0]<< "
            // "<< via[found][1]<<"\n";
        }
    }

    // logger.info("score reduce = %f", reduce);
    // logger.info("move count = %d", move);
    // logger.info("total node = %d", block);
}

//---------------------------------------------------------------------

void post_process(PlaceData& data,
                  torch::Tensor node_pos,
                  torch::Tensor node_die,
                  torch::Tensor via_pos,
                  torch::Tensor via_size,
                  torch::Tensor row_shift,
                  torch::Tensor numRows,
                  torch::Tensor row_height) {
    torch::Tensor pin_id2node_id = data.pin_id2node_id;
    torch::Tensor hyperedge_list = data.hyperedge_list;
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos;

    // std::cout << node_pos <<"\n";

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;          // x, y
    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt));
    auto boundary = torch::zeros({2, num_nets, num_channels, 2}, torch::dtype(pin_pos.dtype()));
    auto via = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));
    auto skip = torch::ones({num_nets}, torch::dtype(torch::kInt));
    // torch::Tensor row_shift = data.die_info[2];

    AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                                   node_pos_to_pin_pos_cross_chip_kernel<scalar_t>(
                                       node_pos.accessor<scalar_t, 2>(),
                                       node_die.accessor<int, 1>(),
                                       pin_id2node_id.accessor<int64_t, 1>(),
                                       pin_pos.accessor<scalar_t, 2>(),
                                       pin_die.accessor<int, 1>(),
                                       num_pins,
                                       at::get_num_threads());
                               }));

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
                              evaluate_boundary<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
                                                          pin_die.accessor<int, 1>(),
                                                          hyperedge_list.accessor<int64_t, 1>(),
                                                          hyperedge_list_end.accessor<int64_t, 1>(),
                                                          boundary.accessor<scalar_t, 4>(),
                                                          via.accessor<scalar_t, 2>(),
                                                          skip.accessor<int, 1>(),
                                                          num_nets,
                                                          at::get_num_threads());
                          }));

    torch::Tensor via_clone = via.clone();

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
                              fill_space<scalar_t>(num_nets,
                                                   boundary.accessor<scalar_t, 4>(),
                                                   via.accessor<scalar_t, 2>(),
                                                   via_pos,
                                                   skip.accessor<int, 1>(),
                                                   numRows,
                                                   row_height,
                                                   row_shift,
                                                   via_size,
                                                   data.core_info,
                                                   num_nets,
                                                   at::get_num_threads());
                          }));
    via = via_pos.clone();
    int via_count = 0;
    ///*
    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
                              swap_via_overlap_region<scalar_t>(num_nets,
                                                                boundary.accessor<scalar_t, 4>(),
                                                                via.accessor<scalar_t, 2>(),
                                                                via_pos,
                                                                skip.accessor<int, 1>(),
                                                                numRows,
                                                                row_height,
                                                                row_shift,
                                                                via_size,
                                                                num_nets,
                                                                at::get_num_threads());
                          }));

    //*/

}  // END MODULE

template <typename scalar_t>
void place_centre(int max_loop,
                  torch::TensorAccessor<scalar_t, 4> boundary,
                  torch::TensorAccessor<scalar_t, 2> via,
                  torch::Tensor via_pos,
                  torch::TensorAccessor<int, 1> skip,
                  torch::Tensor numRows,
                  torch::Tensor row_height,
                  torch::Tensor row_shift,
                  torch::Tensor size,
                  torch::Tensor die_info,
                  int num_nets,
                  int num_threads) {
    int edge[2][2];
    edge[0][0] = die_info[0].item<int>();
    edge[0][1] = die_info[1].item<int>();
    edge[1][0] = die_info[2].item<int>();
    edge[1][1] = die_info[3].item<int>();
    for (int i = 0; i < num_nets; i++) {
        int pos[2];
        for (int axis = 0; axis < 2; axis++) {
            if (boundary[0][i][axis][0] <= boundary[1][i][axis][0] &&
                boundary[0][i][axis][1] >= boundary[1][i][axis][1]) {
                via_pos[i][axis] = (boundary[0][i][axis][0] + boundary[0][i][axis][1]) / 2;
            } else if (boundary[1][i][axis][0] <= boundary[0][i][axis][0] &&
                       boundary[1][i][axis][1] >= boundary[0][i][axis][1]) {
                via_pos[i][axis] = (boundary[1][i][axis][0] + boundary[1][i][axis][1]) / 2;
            } else {
                if (abs(boundary[0][i][axis][0] - boundary[1][i][axis][1]) >=
                    abs(boundary[1][i][axis][0] - boundary[0][i][axis][1])) {
                    via_pos[i][axis] = (boundary[1][i][axis][0] + boundary[0][i][axis][1]) / 2;
                } else {
                    via_pos[i][axis] = (boundary[0][i][axis][0] + boundary[1][i][axis][1]) / 2;
                }
            }
            if (via_pos[i][axis].item<int>() + size[i][axis].item<int>() / 2 > edge[axis][1]) {
                via_pos[i][axis] = edge[axis][1] - size[i][axis].item<int>() / 2;
            } else if (via_pos[i][axis].item<int>() - size[i][axis].item<int>() / 2 < edge[axis][0]) {
                via_pos[i][axis] = edge[axis][0] + size[i][axis].item<int>() / 2;
            }
        }
    }
}

void viaPlaceCentre(PlaceData& data,
                    torch::Tensor node_pos,
                    torch::Tensor node_die,
                    torch::Tensor via_pos,
                    torch::Tensor via_size,
                    torch::Tensor row_shift,
                    torch::Tensor numRows,
                    torch::Tensor row_height) {
    torch::Tensor pin_id2node_id = data.pin_id2node_id;
    torch::Tensor hyperedge_list = data.hyperedge_list;
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos;

    // std::cout << node_pos <<"\n";

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;          // x, y
    auto pin_pos = pin_rel_cpos.clone();  // pin TODO:
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt));
    auto boundary = torch::zeros({2, num_nets, num_channels, 2}, torch::dtype(pin_pos.dtype()));
    auto via = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()));
    auto skip = torch::ones({num_nets}, torch::dtype(torch::kInt));
    // torch::Tensor row_shift = data.die_info[2];

    AT_DISPATCH_FLOATING_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cross_chip", ([&] {
                                   node_pos_to_pin_pos_cross_chip_kernel<scalar_t>(
                                       node_pos.accessor<scalar_t, 2>(),
                                       node_die.accessor<int, 1>(),
                                       pin_id2node_id.accessor<int64_t, 1>(),
                                       pin_pos.accessor<scalar_t, 2>(),
                                       pin_die.accessor<int, 1>(),
                                       num_pins,
                                       at::get_num_threads());
                               }));

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
                              evaluate_boundary<scalar_t>(pin_pos.accessor<scalar_t, 2>(),
                                                          pin_die.accessor<int, 1>(),
                                                          hyperedge_list.accessor<int64_t, 1>(),
                                                          hyperedge_list_end.accessor<int64_t, 1>(),
                                                          boundary.accessor<scalar_t, 4>(),
                                                          via.accessor<scalar_t, 2>(),
                                                          skip.accessor<int, 1>(),
                                                          num_nets,
                                                          at::get_num_threads());
                          }));

    AT_DISPATCH_ALL_TYPES(pin_pos.scalar_type(), "hpwl", ([&] {
                              place_centre<scalar_t>(num_nets,
                                                     boundary.accessor<scalar_t, 4>(),
                                                     via.accessor<scalar_t, 2>(),
                                                     via_pos,
                                                     skip.accessor<int, 1>(),
                                                     numRows,
                                                     row_height,
                                                     row_shift,
                                                     via_size,
                                                     data.core_info,
                                                     num_nets,
                                                     at::get_num_threads());
                          }));

}  // END MODULE

//---------------------------------------------------------------------