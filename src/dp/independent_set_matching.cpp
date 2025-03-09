#include "independent_set_matching.h"

#include <omp.h>

namespace dp {

int ceil_power2(int v) { return (1 << (int)ceil(log2(v))); }

void construct_spaces(DetailedPlaceData& db,
                      const torch::TensorAccessor<float, 1> host_x,
                      const torch::TensorAccessor<float, 1> host_y,
                      std::vector<Space>& host_spaces,
                      int num_threads) {
    std::vector<std::vector<int>> row2node_map(db.num_sites_y);
    db.make_row2node_map(host_x, host_y, row2node_map);

    // construct spaces
    host_spaces.resize(db.i_end);
    for (int i = 0; i < db.num_sites_y; ++i) {
        for (unsigned int j = 0; j < row2node_map[i].size(); ++j) {
            auto const& row2nodes = row2node_map[i];
            int node_id = row2nodes[j];
            auto& space = host_spaces[node_id];
            if (node_id < db.i_end) {
                auto left_bound = db.xl;
                if (j) {
                    left_bound = host_x[node_id];
                }
                space.xl = left_bound;

                auto right_bound = db.xh;
                if (j + 1 < row2nodes.size()) {
                    int right_node_id = row2nodes[j + 1];
                    right_bound = std::min(right_bound, host_x[right_node_id]);
                }
                space.xh = right_bound;
            }
        }
    }
}

void mark_dependent_nodes(const DetailedPlaceData& db,
                          IndependentSetMatchingState& state,
                          int node_id,
                          unsigned char value) {
    if (db.node_weight[node_id] == 0) return;
    float node_xl = db.x[node_id];
    float node_yl = db.y[node_id];
    // in case all nets are masked
    int node2pin_start = db.flat_node2pin_start_map[node_id];
    int node2pin_end = db.flat_node2pin_start_map[node_id + 1];
    for (int node2pin_id = node2pin_start; node2pin_id < node2pin_end; ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        if (db.net_mask[net_id]) {
            int net2pin_start = db.flat_net2pin_start_map[net_id];
            int net2pin_end = db.flat_net2pin_start_map[net_id + 1];
            for (int net2pin_id = net2pin_start; net2pin_id < net2pin_end; ++net2pin_id) {
                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                int other_node_id = db.pin2node_map[net_pin_id];
                if (db.node_weight[other_node_id] == 0) continue;
                float other_node_xl = db.x[other_node_id];
                float other_node_yl = db.y[other_node_id];
                if (std::abs(node_xl - other_node_xl) + std::abs(node_yl - other_node_yl) < state.skip_threshold) {
                    if (other_node_id < db.num_nodes) {
                        if (db.node_weight[other_node_id] == 0 || other_node_id >= db.i_end)
                            continue;  // FIXME
                        state.dependent_markers[other_node_id] = value;
                    }
                }
            }
        }
    }
    state.dependent_markers[node_id] = value;
}

void maximal_independent_set_sequential(const DetailedPlaceData& db, IndependentSetMatchingState& state) {
    std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);
    std::fill(state.dependent_markers.begin(), state.dependent_markers.end(), 0);

    for (int i = db.i_bgn; i < db.i_end; ++i) {
        int node_id = state.ordered_nodes[i];
        if (db.node_weight[node_id] == 0) continue;
        if (!state.dependent_markers[node_id]) {
            state.selected_markers[node_id] = 1;
            mark_dependent_nodes(db, state, node_id, 1);
        }
    }
}

void construct_selected_node2bin_map(const DetailedPlaceData& db, IndependentSetMatchingState& state) {
    for (auto& bin2nodes : state.bin2node_map) {
        bin2nodes.clear();
    }
    for (int node_id = db.i_bgn; node_id < db.i_end; ++node_id) {
        if (db.node_weight[node_id] == 0) continue;
        if (state.selected_markers[node_id]) {
            float width = db.node_size_x[node_id];
            float height = db.node_size_y[node_id];
            auto& bm_idx = state.node2bin_map[node_id];
            int num_bins_x = db.num_bins_x;
            int num_bins_y = db.num_bins_y;
            float bin_size_x = db.bin_size_x;
            float bin_size_y = db.bin_size_y;

            float node_x = db.x[node_id] + width / 2;
            float node_y = db.y[node_id] + height / 2;

            int bx = std::min(std::max((int)floorDiv((node_x - db.xl), bin_size_x), 0), num_bins_x - 1);
            int by = std::min(std::max((int)floorDiv((node_y - db.yl), bin_size_y), 0), num_bins_y - 1);
            bm_idx.bin_id = bx * num_bins_y + by;

            auto& bin2nodes = state.bin2node_map.at(bm_idx.bin_id);
            bin2nodes.push_back(node_id);
        }
    }
}

int partitioning_diamond(const DetailedPlaceData& db, IndependentSetMatchingState& state) {
    // assume cells have been distributed to bins
    state.independent_sets.resize(state.batch_size);
    state.solutions.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    int num_independent_sets = 0;
    for (int i = db.i_bgn; i < db.i_end; ++i) {
        int seed_node = state.ordered_nodes.at(i);
        if (state.selected_markers.at(seed_node)) {
            if (db.node_weight[seed_node] == 0) continue;
            float seed_height = db.node_size_y[seed_node];
            auto const& seed_bin = state.node2bin_map.at(seed_node);
            int num_bins_x = db.num_bins_x;
            int num_bins_y = db.num_bins_y;
            int seed_bin_x = seed_bin.bin_id / num_bins_y;
            int seed_bin_y = seed_bin.bin_id % num_bins_y;
            auto const& bin2node_map = state.bin2node_map;
            auto& independent_set = state.independent_sets.at(num_independent_sets);
            ++num_independent_sets;
            independent_set.clear();
            for (int j = 0; j < state.max_diamond_search_sequence; ++j) {
                // get bin (bx, by)
                int bx = seed_bin_x + state.search_grids.at(j).ic;
                int by = seed_bin_y + state.search_grids.at(j).ir;
                if (bx < 0 || bx >= num_bins_x || by < 0 || by >= num_bins_y) {
                    continue;
                }
                int bin_id = bx * num_bins_y + by;
                auto const& bin2nodes = bin2node_map.at(bin_id);

                for (auto node_id : bin2nodes) {
                    if (db.node_size_y[node_id] == seed_height && state.selected_markers.at(node_id)) {
                        independent_set.push_back(node_id);
                        state.selected_markers.at(node_id) = 0;
                        if (independent_set.size() >= (unsigned int)state.set_size) {
                            break;
                        }
                    }
                }
                if (independent_set.size() >= (unsigned int)state.set_size) {
                    break;
                }
            }
            // make sure batch_size is large enough
            if (num_independent_sets >= state.batch_size) {
                break;
            }
        }
    }

    return num_independent_sets;
}

int collect_independent_sets(const DetailedPlaceData& db, IndependentSetMatchingState& state) {
    construct_selected_node2bin_map(db, state);
    for (auto& independent_set : state.independent_sets) {
        independent_set.clear();
    }

    int num_independent_sets = partitioning_diamond(db, state);

    // sort sets according to large to small
    std::sort(state.independent_sets.begin(),
              state.independent_sets.end(),
              [&](const std::vector<int>& s1, const std::vector<int>& s2) { return s1.size() > s2.size(); });
    // clean small sets
    for (int i = 0; i < (int)state.independent_sets.size(); ++i) {
        if (i >= state.batch_size || state.independent_sets.at(i).size() < 3U) {
            state.independent_sets.at(i).clear();
        } else {
            num_independent_sets = i;
        }
    }
    // shrink large sets
    for (auto& independent_set : state.independent_sets) {
        if (independent_set.size() > (unsigned int)state.set_size) {
            independent_set.resize(state.set_size);
        }
    }

    int avg_set_size = 0;
    int max_set_size = 0;
    for (int i = 0; i < num_independent_sets; ++i) {
        avg_set_size += state.independent_sets.at(i).size();
        max_set_size = std::max(max_set_size, (int)state.independent_sets.at(i).size());
    }
    // logger.debug("%d sets, average set size %d, max set size %d",
    //              num_independent_sets,
    //              avg_set_size / num_independent_sets,
    //              max_set_size);

    return num_independent_sets;
}

bool adjust_pos(float& x, float width, const Space& space) {
    // the order is very tricky for numerical stability
    x = std::min(x, space.xh - width);
    x = std::max(x, space.xl);
    return width + space.xl <= space.xh;
}

void cost_matrix_construction(const DetailedPlaceData& db,
                              IndependentSetMatchingState& state,
                              bool major,  ///< false: row major, true: column major
                              int i        ///< entry in the batch
) {
    auto const& independent_set = state.independent_sets[i];
    auto& cost_matrix = state.cost_matrices[i];
    unsigned int independent_set_size = independent_set.size();
    std::vector<Box> bboxes;
    // cells
    for (unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        float node_width = db.node_size_x[node_id];
        bboxes.resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            Box& box = bboxes[idx];
            box.xl = db.xh;
            box.yl = db.yh;
            box.xh = db.xl;
            box.yh = db.yl;
            if (db.net_mask[net_id]) {
                for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                     net2pin_id < db.flat_net2pin_start_map[net_id + 1];
                     ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (db.node_die[other_node_id] == 0) continue;
                    if (other_node_id != node_id) {
                        float xxl = db.x[other_node_id];
                        float yyl = db.y[other_node_id];
                        box.xl = std::min(box.xl, xxl + db.pin_offset_x[net_pin_id]);
                        box.xh = std::max(box.xh, xxl + db.pin_offset_x[net_pin_id]);
                        box.yl = std::min(box.yl, yyl + db.pin_offset_y[net_pin_id]);
                        box.yh = std::max(box.yh, yyl + db.pin_offset_y[net_pin_id]);
                    }
                }
            }
        }
        for (unsigned int j = 0; j < independent_set_size; ++j) {
            int pos_id = independent_set[j];
            if (db.node_weight[pos_id] == 0) continue;
            float target_x = db.x[pos_id];
            float target_y = db.y[pos_id];
            float target_hpwl = 0;

            auto const& target_space = state.spaces[pos_id];
            if (adjust_pos(target_x, node_width, target_space)) {
                // adjust_pos(target_x, db.node_size_x[pos_id], target_space);
                //  consider FENCE region

                idx = 0;
                for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                     node2pin_id < db.flat_node2pin_start_map[node_id + 1];
                     ++node2pin_id, ++idx) {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    const Box& box = bboxes[idx];
                    if (db.net_mask[net_id]) {
                        float xxl = target_x;
                        float yyl = target_y;
                        float bxl = std::min(box.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh = std::max(box.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl = std::min(box.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh = std::max(box.yh, yyl + db.pin_offset_y[node_pin_id]);
                        target_hpwl += (bxh - bxl) + (byh - byl);
                    }
                }

            } else {
                // dreamplaceAssertMsg(node_id != pos_id, "node %d, pos %d", node_id, pos_id);
                target_hpwl = state.large_number;
            }

            if (!major)  // row major
            {
                cost_matrix[independent_set.size() * k + j] = target_hpwl;
            } else  // column major
            {
                cost_matrix[independent_set.size() * j + k] = target_hpwl;
            }
        }
    }
}

void cost_matrix_construction_via(const DetailedPlaceData& db,
                              IndependentSetMatchingState& state,
                              bool major,  ///< false: row major, true: column major
                              int i        ///< entry in the batch
) {
    auto const& independent_set = state.independent_sets[i];
    auto& cost_matrix = state.cost_matrices[i];
    unsigned int independent_set_size = independent_set.size();
    std::vector<std::vector<Box>> bboxes;
    bboxes.resize(2);
    // cells
    for (unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        float node_width = db.node_size_x[node_id];
        bboxes[0].resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        bboxes[1].resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];

            Box& box0 = bboxes[0][idx];
            Box& box1 = bboxes[1][idx];

            box0.xl = db.xh;
            box0.yl = db.yh;
            box0.xh = db.xl;
            box0.yh = db.yl;
            box1.xl = db.xh;
            box1.yl = db.yh;
            box1.xh = db.xl;
            box1.yh = db.yl;
            if (db.net_mask[net_id]) {
                for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                     net2pin_id < db.flat_net2pin_start_map[net_id + 1];
                     ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (db.node_die[other_node_id] == -1) continue;
                    if (other_node_id != node_id) {
                        int c_id = db.node_die[other_node_id];
                        float xxl = db.x[other_node_id];
                        float yyl = db.y[other_node_id];
                        if (c_id == 0) {
                            box0.xl = std::min(box0.xl, xxl + db.pin_offset_x[net_pin_id]);
                            box0.xh = std::max(box0.xh, xxl + db.pin_offset_x[net_pin_id]);
                            box0.yl = std::min(box0.yl, yyl + db.pin_offset_y[net_pin_id]);
                            box0.yh = std::max(box0.yh, yyl + db.pin_offset_y[net_pin_id]);
                        } else if (c_id == 1) {
                            box1.xl = std::min(box1.xl, xxl + db.pin_offset_x[net_pin_id]);
                            box1.xh = std::max(box1.xh, xxl + db.pin_offset_x[net_pin_id]);
                            box1.yl = std::min(box1.yl, yyl + db.pin_offset_y[net_pin_id]);
                            box1.yh = std::max(box1.yh, yyl + db.pin_offset_y[net_pin_id]);
                        }
                        
                        else {
                            box0.xl = std::min(box0.xl, xxl);
                            box0.xh = std::max(box0.xh, xxl);
                            box0.yl = std::min(box0.yl, yyl);
                            box0.yh = std::max(box0.yh, yyl);

                            box1.xl = std::min(box1.xl, xxl);
                            box1.xh = std::max(box1.xh, xxl);
                            box1.yl = std::min(box1.yl, yyl);
                            box1.yh = std::max(box1.yh, yyl);
                        }
    
                        

                    }
                }
            }

            // cout << " ------1----- \n";
            // cout << box0.xl << endl;
            // cout << box1.xl << endl;
            // cout << box0.yl << endl;
            // cout << box1.yl << endl;
        }

        for (unsigned int j = 0; j < independent_set_size; ++j) {
            int pos_id = independent_set[j];
            if (db.node_weight[pos_id] == 0) continue;
            float target_x = db.x[pos_id];
            float target_y = db.y[pos_id];
            float target_hpwl = 0;

            auto const& target_space = state.spaces[pos_id];
            if (adjust_pos(target_x, node_width, target_space)) {
                // adjust_pos(target_x, db.node_size_x[pos_id], target_space);
                //  consider FENCE region

                idx = 0;
                for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                     node2pin_id < db.flat_node2pin_start_map[node_id + 1];
                     ++node2pin_id, ++idx) {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];

                    const Box& box0 = bboxes[0][idx];
                    const Box& box1 = bboxes[1][idx];

                    // cout << " ------2------ \n";
                    // cout << box0.xl << endl;
                    // cout << box1.xl << endl;
                    // cout << box0.yl << endl;
                    // cout << box1.yl << endl;
                    
                    if (db.net_mask[net_id]) {
                        float xxl = target_x;
                        float yyl = target_y;
                        float bxl0 = std::min(box0.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh0 = std::max(box0.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl0 = std::min(box0.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh0 = std::max(box0.yh, yyl + db.pin_offset_y[node_pin_id]);
                        float bxl1 = std::min(box1.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh1 = std::max(box1.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl1 = std::min(box1.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh1 = std::max(box1.yh, yyl + db.pin_offset_y[node_pin_id]);
                        target_hpwl += (bxh0 - bxl0) + (byh0 - byl0) + (bxh1 - bxl1) + (byh1 - byl1);


                    }

                    // cout << box0.xl << endl;
                    // cout << box1.xl << endl;
                    // cout << box0.yl << endl;
                    // cout << box1.yl << endl;

                    // cout << target_hpwl << endl;
                }
            } else {
                // dreamplaceAssertMsg(node_id != pos_id, "node %d, pos %d", node_id, pos_id);
                target_hpwl = state.large_number;
            }

            if (!major)  // row major
            {
                cost_matrix[independent_set.size() * k + j] = target_hpwl;
            } else  // column major
            {
                cost_matrix[independent_set.size() * j + k] = target_hpwl;
            }
        }
    }
}


void apply_solution(DetailedPlaceData& db, IndependentSetMatchingState& state, int i) {
    auto const& independent_set = state.independent_sets.at(i);
    auto& solution = state.solutions.at(i);
    auto& target_pos_x = state.target_pos_x.at(i);
    auto& target_pos_y = state.target_pos_y.at(i);
    auto& target_spaces = state.target_spaces.at(i);
    target_pos_x.resize(independent_set.size());
    target_pos_y.resize(independent_set.size());
    target_spaces.resize(independent_set.size());
    /* FIXME some unknown bugs occured in some small independent_sets
    if( independent_set.size()<(unsigned int)state.set_size )
    {
        return;
    }
    */

    // apply solution
    if (state.target_costs[i] < state.orig_costs[i]) {
        // record the locations
        for (unsigned int j = 0; j < independent_set.size(); ++j) {
            int target_node_id = independent_set.at(j);
            if (target_node_id < db.i_end) {
                if (db.node_weight[target_node_id] == 0) continue;
                target_pos_x[j] = db.x[target_node_id];
                target_pos_y[j] = db.y[target_node_id];
                target_spaces[j] = state.spaces[target_node_id];
            }
        }
        // move cells
        int count = 0;
        for (unsigned int j = 0; j < independent_set.size(); ++j) {
            int sol_j = solution.at(j);
            int target_node_id = independent_set.at(j);//independent_set -> solution   independent_set是原集, solution是对应的解
            if (target_node_id < db.i_end) {
                if (db.node_weight[target_node_id] == 0) continue;
                int target_pos_id = independent_set.at(sol_j);
                assert(target_pos_id < db.i_end);
                //#endif
                if (j != (unsigned int)sol_j) {
                    count += 1;
                    bool ret = adjust_pos(target_pos_x[sol_j], db.node_size_x[target_node_id], target_spaces[sol_j]);
                    assert_msg(ret,
                               "set %d (%lu nodes), node %d, width %g, %g, %g, pos %d, %g, %g, space %g, %g, "
                               "orig_cost %d, target_cost %d, cost %d",
                               i,
                               independent_set.size(),
                               target_node_id,
                               db.node_size_x[target_node_id],//width
                               db.x[target_node_id],
                               db.x[target_node_id] + db.node_size_x[target_node_id],
                               target_pos_id,//pos
                               target_pos_x[sol_j],
                               target_pos_x[sol_j] + db.node_size_x[target_pos_id],
                               target_spaces[sol_j].xl,//space
                               target_spaces[sol_j].xh,
                               state.orig_costs[i],//orig_cost
                               state.target_costs[i],//target_cost
                               state.cost_matrices.at(i).at(j * independent_set.size() + sol_j));//cost
                    // update position
                    db.x[target_node_id] = target_pos_x[sol_j];
                    db.y[target_node_id] = target_pos_y[sol_j];
                    state.spaces.at(target_node_id) = target_spaces[sol_j];
                }
            }
        }
    }
}

void independentSetMatching(
    DetailedPlaceData& db, int num_bins_x, int num_bins_y, int batch_size, int set_size, int max_iters) {
    logger.info("============= Running DP: %d-ISM =============", max_iters);
    db.num_bins_x = num_bins_x;
    db.num_bins_y = num_bins_y;
    db.bin_size_x = (db.xh - db.xl) / num_bins_x;
    db.bin_size_y = (db.yh - db.yl) / num_bins_y;
    // const double threshold = 0.00001/100;
    IndependentSetMatchingState state;
    state.batch_size = batch_size;
    state.set_size = set_size;
    state.num_moved = 0;
    state.large_number = (db.xh - db.xl + db.yh - db.yl) * set_size;
    state.skip_threshold = (db.xh - db.xl + db.yh - db.yl) * 0.01;
    state.num_threads = std::max(db.num_threads, 1);

    state.bin2node_map.resize(db.num_bins_x * db.num_bins_y);
    state.node2bin_map.resize(db.i_end);
    // make_bin2node_map(db, db.x, db.y, db.node_size_x, db.node_size_y, state);
    construct_spaces(db, db.x, db.y, state.spaces, state.num_threads);
    state.grid_size = ceil_power2(std::max(db.num_bins_x, db.num_bins_y) / 8);
    state.max_diamond_search_sequence = state.grid_size * state.grid_size / 2;
    logger.info("diamond search grid size %d, sequence length %d", state.grid_size, state.max_diamond_search_sequence);

    state.ordered_nodes.resize(db.i_end);
    std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);
    state.independent_sets.resize(state.batch_size, std::vector<int>(state.set_size));
    state.dependent_markers.assign(db.num_nodes, 0);
    state.selected_markers.assign(db.i_end, 0);
    state.num_selected_markers.assign(db.i_end, 0);
    state.search_grids = diamond_search_sequence(state.grid_size, state.grid_size);

    state.cost_matrices.resize(state.batch_size);
    state.solutions.resize(state.batch_size);
    state.orig_costs.resize(state.batch_size);
    state.target_costs.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    state.target_spaces.resize(state.batch_size);
    std::vector<AuctionAlgorithmCPULauncher<int>> solvers(state.num_threads);

    bool major = false;  // row major

    std::vector<float> hpwls(max_iters + 1);
    hpwls.at(0) = db.compute_total_hpwl();
    logger.info("initial hpwl %g", hpwls.at(0));
    for (int iter = 0; iter < max_iters; ++iter) {
        // std::random_shuffle(state.ordered_nodes.begin(),
        // state.ordered_nodes.end());
        std::sort(state.ordered_nodes.begin(), state.ordered_nodes.end(), [&](int node_id1, int node_id2) {
            return state.num_selected_markers.at(node_id1) < state.num_selected_markers.at(node_id2);
        });
        std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);

        // for small benchmarks, sequential version is faster
        // as the parallel algorithm needs to run at most 10 times,
        // there will be no benefit with 10 or fewer threads
        // FIXME: we remove paralle mode
        // if (state.num_threads < 10) {
        //     maximal_independent_set_sequential(db, state);
        // } else {
        //     maximal_independent_set_parallel(db, state);
        // }
        maximal_independent_set_sequential(db, state);

        int num_independent_sets = collect_independent_sets(db, state);
#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < num_independent_sets; ++i) {
            for (auto node_id : state.independent_sets.at(i)) {
                if (node_id < db.i_end) {
                    if (db.node_weight[node_id] == 0) continue;
                    state.num_selected_markers.at(node_id) += 1;
                }
            }
        }

        if (num_independent_sets > state.batch_size) {
            state.cost_matrices.resize(num_independent_sets);
            state.solutions.resize(num_independent_sets);
            state.orig_costs.resize(num_independent_sets);
            state.target_costs.resize(num_independent_sets);
            state.target_pos_x.resize(num_independent_sets);
            state.target_pos_y.resize(num_independent_sets);
            state.target_spaces.resize(num_independent_sets);
        }

#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < num_independent_sets; ++i) {
            auto const& independent_set = state.independent_sets.at(i);
            auto& cost_matrix = state.cost_matrices.at(i);
            cost_matrix.resize(independent_set.size() * independent_set.size());

            if (!db.via_dp) {
                cost_matrix_construction(db, state, major, i);
            } else {
                cost_matrix_construction_via(db, state, major, i);
            }
        }

#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < num_independent_sets; ++i) {
            auto const& independent_set = state.independent_sets.at(i);
            auto& cost_matrix = state.cost_matrices.at(i);
            auto& solution = state.solutions.at(i);
            auto& orig_cost = state.orig_costs.at(i);
            auto& target_cost = state.target_costs.at(i);
            solution.resize(independent_set.size());

            // solve bipartite assignment problem
            // compute initial cost
            orig_cost = 0;
            for (unsigned int j = 0; j < independent_set.size(); ++j) {
                orig_cost += cost_matrix.at(j * independent_set.size() + j);
            }
            int tid = omp_get_thread_num();
            target_cost = solvers.at(tid).run(cost_matrix.data(), solution.data(), independent_set.size());//find a solution
        }

#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < num_independent_sets; ++i) {
            apply_solution(db, state, i);
        }

        hpwls.at(iter + 1) = db.compute_total_hpwl();
        if ((iter % (std::max(max_iters / 10, 1))) == 0 || iter + 1 == max_iters) {
            state.num_moved = 0;
            for (int i = 0; i < db.i_end; ++i) {
                if (db.node_weight[i] == 0) continue;
                if (db.x[i] != db.init_x[i] || db.y[i] != db.init_y[i]) {
                    state.num_moved += 1;
                }
            }
            logger.info(
                "iteration %d, target hpwl %g, delta %g(%g%%), solved %d sets, moved "
                "%g%% cells",
                iter,
                hpwls.at(iter + 1),
                hpwls.at(iter + 1) - hpwls.at(0),
                (hpwls.at(iter + 1) - hpwls.at(0)) / hpwls.at(0) * 100,
                num_independent_sets,
                state.num_moved / (double)db.i_end * 100);
        }

        // if (iter && hpwls.at(iter)-hpwls.at(iter+1) < threshold*hpwls.at(iter))
        //{
        //    break;
        //}
    }
}
// =================================================================

void make_bin2node_map(const DetailedPlaceData& db,
                       torch::TensorAccessor<float, 1> host_x,
                       torch::TensorAccessor<float, 1> host_y,
                       torch::TensorAccessor<float, 1> host_node_size_x,
                       torch::TensorAccessor<float, 1> host_node_size_y,
                       IndependentSetMatchingState& state) {
    // construct bin2node_map
    state.bin2node_map.resize(db.num_bins_x * db.num_bins_y);
    for (int i = 0; i < db.i_end; ++i) {
        if (db.node_weight[i] == 0) continue;
        int node_id = i;
        float node_x = host_x[node_id] + host_node_size_x[node_id] / 2;
        float node_y = host_y[node_id] + host_node_size_y[node_id] / 2;

        int bx = std::min(std::max((int)floorDiv((node_x - db.xl), db.bin_size_x), 0), db.num_bins_x - 1);
        int by = std::min(std::max((int)floorDiv((node_y - db.yl), db.bin_size_y), 0), db.num_bins_y - 1);
        int bin_id = bx * db.num_bins_y + by;
        // int sub_id = bin2node_map.at(bin_id).size();
        state.bin2node_map.at(bin_id).push_back(node_id);
    }
    // construct node2bin_map
    state.node2bin_map.resize(db.i_end);
    for (unsigned int bin_id = 0; bin_id < state.bin2node_map.size(); ++bin_id) {
        for (unsigned int sub_id = 0; sub_id < state.bin2node_map[bin_id].size(); ++sub_id) {
            int node_id = state.bin2node_map[bin_id][sub_id];
            BinMapIndex& bm_idx = state.node2bin_map.at(node_id);
            bm_idx.bin_id = bin_id;
            bm_idx.sub_id = sub_id;
        }
    }
}

bool collect_independent_sets_sequential(const DetailedPlaceData& db,
                                         IndependentSetMatchingState& state,
                                         int seed_node,
                                         int i  ///< entry in batch
) {
    auto& independent_set = state.independent_sets[i];
    independent_set.clear();

    float seed_height = db.node_size_y[seed_node];
    auto const& seed_bin = state.node2bin_map.at(seed_node);
    int num_bins_x = db.num_bins_x;
    int num_bins_y = db.num_bins_y;
    int seed_bin_x = seed_bin.bin_id / num_bins_y;
    int seed_bin_y = seed_bin.bin_id % num_bins_y;
    // int seed_bin_id = seed_bin_x*num_bins_y + seed_bin_y;
    auto const& bin2node_map = state.bin2node_map;

    for (int j = 0; j < state.max_diamond_search_sequence; ++j) {
        // get bin (bx, by)
        int bx = seed_bin_x + state.search_grids[j].ic;
        int by = seed_bin_y + state.search_grids[j].ir;
        if (bx < 0 || bx >= num_bins_x || by < 0 || by >= num_bins_y) {
            continue;
        }
        int bin_id = bx * num_bins_y + by;

        auto const& bin2nodes = bin2node_map.at(bin_id);

        for (auto node_id : bin2nodes) {
            if (db.node_size_y[node_id] == seed_height && !state.dependent_markers[node_id]) {
                independent_set.push_back(node_id);
                mark_dependent_nodes(db, state, node_id, 1);
                state.selected_markers[node_id] = 1;
                state.num_selected_markers[node_id] += 1;
                if (independent_set.size() >= (unsigned int)state.set_size) {
                    break;
                }
            }
        }
        if (independent_set.size() >= (unsigned int)state.set_size) {
            break;
        }
    }

    for (auto node_id : independent_set) {
        mark_dependent_nodes(db, state, node_id, 0);
    }

    return true;
}

void apply_solution_sequential(DetailedPlaceData& db,
                               IndependentSetMatchingState& state,
                               int i  ///< entry in the batch
) {
    auto const& independent_set = state.independent_sets.at(i);
    auto& solution = state.solutions.at(i);
    auto& target_pos_x = state.target_pos_x.at(i);
    auto& target_pos_y = state.target_pos_y.at(i);
    auto& target_node2bin_map = state.target_node2bin_map.at(i);
    auto& target_spaces = state.target_spaces.at(i);
    solution.resize(independent_set.size());
    target_pos_x.resize(independent_set.size());
    target_pos_y.resize(independent_set.size());
    target_node2bin_map.resize(independent_set.size());
    target_spaces.resize(independent_set.size());

    // apply solution
    if (state.target_costs[i] < state.orig_costs[i]) {
        // record the locations
        for (unsigned int j = 0; j < independent_set.size(); ++j) {
            int target_node_id = independent_set.at(j);
            if (target_node_id < db.i_end) {
                if (db.node_weight[target_node_id] == 0) continue;
                target_pos_x[j] = db.x[target_node_id];
                target_pos_y[j] = db.y[target_node_id];
                target_node2bin_map[j] = state.node2bin_map[target_node_id];
                target_spaces[j] = state.spaces[target_node_id];
            }
        }
        // move cells
        int count = 0;
        for (unsigned int j = 0; j < independent_set.size(); ++j) {
            int sol_j = solution.at(j);
            int target_node_id = independent_set.at(j);
            if (target_node_id < db.i_end) {
                if (db.node_weight[target_node_id] == 0) continue;
                if (db.x[target_node_id] != target_pos_x[sol_j] || db.y[target_node_id] != target_pos_y[sol_j]) {
                    count += 1;
                }
                // update position
                adjust_pos(target_pos_x[sol_j], db.node_size_x[target_node_id], target_spaces[sol_j]);
                db.x[target_node_id] = target_pos_x[sol_j];
                db.y[target_node_id] = target_pos_y[sol_j];
                auto const& bm_idx = target_node2bin_map[sol_j];
                state.bin2node_map.at(bm_idx.bin_id).at(bm_idx.sub_id) = target_node_id;
                state.node2bin_map[target_node_id] = bm_idx;
                state.spaces.at(target_node_id) = target_spaces[sol_j];
            }
        }
        //#pragma omp atomic
        state.num_moved += count;
    }
}

void cost_matrix_construction2(const DetailedPlaceData& db,
                              IndependentSetMatchingState& state,
                              bool major,  ///< false: row major, true: column major
                              int i        ///< entry in the batch
) {
    auto const& independent_set = state.independent_sets[i];
    auto& cost_matrix = state.cost_matrices[i];
    unsigned int independent_set_size = independent_set.size();
    std::vector<Box> bboxes;
    std::unordered_map<int, NetBoundingBox> netID2bound_map;
    std::unordered_map<int, vector<int>> netID2pinID_map;
    std::unordered_map<int, int> node_id_map;
    for(unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        node_id_map.insert({node_id, 1});
    }
    for(unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            auto got = netID2bound_map.find(net_id);
            if (got != netID2bound_map.end()) continue;
            float xl = db.xh;
            float yl = db.yh;
            float xh = db.xl;
            float yh = db.yl;
            vector<int> permute_node_pin;
            if (db.net_mask[net_id]) {
                for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                     net2pin_id < db.flat_net2pin_start_map[net_id + 1];
                     ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (db.node_die[other_node_id] == 0) continue;
                    if (node_id_map.find(other_node_id) == node_id_map.end()) {
                        float xxl = db.x[other_node_id];
                        float yyl = db.y[other_node_id];
                        xl = std::min(xl, xxl + db.pin_offset_x[net_pin_id]);
                        xh = std::max(xh, xxl + db.pin_offset_x[net_pin_id]);
                        yl = std::min(yl, yyl + db.pin_offset_y[net_pin_id]);
                        yh = std::max(yh, yyl + db.pin_offset_y[net_pin_id]);
                    } else {
                        permute_node_pin.push_back(net_pin_id);
                    }
                    
                }
            }
            NetBoundingBox bound;
            bound.die[0] = Box(xl, yl, xh, yh);
            netID2bound_map.insert({net_id, bound});
            netID2pinID_map.insert({net_id, permute_node_pin});
        }
    }
    // cells

    for (unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        float node_width = db.node_size_x[node_id];
        bboxes.resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        std::unordered_map<int, NetBoundingBox> netID2bound_map_update;
        for(auto kv : netID2pinID_map) {
            int net_id = kv.first;
            Box box(netID2bound_map.at(net_id).die[0].xl,
                    netID2bound_map.at(net_id).die[0].yl,
                    netID2bound_map.at(net_id).die[0].xh, 
                    netID2bound_map.at(net_id).die[0].yh);
            for(int i = 0; i < kv.second.size(); i++) {
                int net_pin_id = kv.second[i];
                int other_node_id = db.pin2node_map[net_pin_id];
                if (db.node_die[other_node_id] == 0) continue;
                if ( (other_node_id != node_id)) {
                    float xxl = db.x[other_node_id];
                    float yyl = db.y[other_node_id];
                    box.xl = std::min(box.xl, xxl + db.pin_offset_x[net_pin_id]);
                    box.xh = std::max(box.xh, xxl + db.pin_offset_x[net_pin_id]);
                    box.yl = std::min(box.yl, yyl + db.pin_offset_y[net_pin_id]);
                    box.yh = std::max(box.yh, yyl + db.pin_offset_y[net_pin_id]);
                }
                
                NetBoundingBox bound;
                bound.die[0] = box;
                netID2bound_map_update.insert({net_id, bound});
            }
        }
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            Box& box = bboxes[idx];
            box.xl = netID2bound_map.at(net_id).die[0].xl;
            box.yl = netID2bound_map.at(net_id).die[0].yl;
            box.xh = netID2bound_map.at(net_id).die[0].xh;
            box.yh = netID2bound_map.at(net_id).die[0].yh;
            
        }
        for (unsigned int j = 0; j < independent_set_size; ++j) {
            int pos_id = independent_set[j];
            if (db.node_weight[pos_id] == 0) continue;
            float target_x = db.x[pos_id];
            float target_y = db.y[pos_id];
            float target_hpwl = 0;

            auto const& target_space = state.spaces[pos_id];
            if (adjust_pos(target_x, node_width, target_space)) {
                // adjust_pos(target_x, db.node_size_x[pos_id], target_space);
                //  consider FENCE region

                idx = 0;
                for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                     node2pin_id < db.flat_node2pin_start_map[node_id + 1];
                     ++node2pin_id, ++idx) {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    const Box& box = bboxes[idx];
                    if (db.net_mask[net_id]) {
                        float xxl = target_x;
                        float yyl = target_y;
                        float bxl = std::min(box.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh = std::max(box.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl = std::min(box.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh = std::max(box.yh, yyl + db.pin_offset_y[node_pin_id]);
                        target_hpwl += (bxh - bxl) + (byh - byl);
                    }
                }

            } else {
                // dreamplaceAssertMsg(node_id != pos_id, "node %d, pos %d", node_id, pos_id);
                target_hpwl = state.large_number;
            }

            if (!major)  // row major
            {
                cost_matrix[independent_set.size() * k + j] = target_hpwl;
            } else  // column major
            {
                cost_matrix[independent_set.size() * j + k] = target_hpwl;
            }
        }
    }
}

void cost_matrix_construction_via2(const DetailedPlaceData& db,
                              IndependentSetMatchingState& state,
                              bool major,  ///< false: row major, true: column major
                              int i        ///< entry in the batch
) {
    auto const& independent_set = state.independent_sets[i];
    auto& cost_matrix = state.cost_matrices[i];
    unsigned int independent_set_size = independent_set.size();
    std::vector<std::vector<Box>> bboxes;
    bboxes.resize(2);
    std::unordered_map<int, NetBoundingBox> netID2bound_map;
    std::unordered_map<int, int> node_id_map;
    std::unordered_map<int, vector<int>> netID2pinID_map;
    for(unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        node_id_map.insert({node_id, 1});
    }
    for(unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            auto got = netID2bound_map.find(net_id);
            if (got != netID2bound_map.end()) continue;
            float xl0 = db.xh;
            float yl0 = db.yh;
            float xh0 = db.xl;
            float yh0 = db.yl;
            float xl1 = db.xh;
            float yl1 = db.yh;
            float xh1 = db.xl;
            float yh1 = db.yl;
            vector<int> permute_node_pin;
            if (db.net_mask[net_id]) {
                for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                     net2pin_id < db.flat_net2pin_start_map[net_id + 1];
                     ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (db.node_die[other_node_id] == -1) continue;
                    if (node_id_map.find(other_node_id) == node_id_map.end()) {
                        float xxl = db.x[other_node_id];
                        float yyl = db.y[other_node_id];
                        int c_id = db.node_die[other_node_id];
                        if (c_id == 0) {
                            xl0 = std::min(xl0, xxl + db.pin_offset_x[net_pin_id]);
                            xh0 = std::max(xh0, xxl + db.pin_offset_x[net_pin_id]);
                            yl0 = std::min(yl0, yyl + db.pin_offset_y[net_pin_id]);
                            yh0 = std::max(yh0, yyl + db.pin_offset_y[net_pin_id]);
                        } else if (c_id == 1) {
                            xl1 = std::min(xl1, xxl + db.pin_offset_x[net_pin_id]);
                            xh1 = std::max(xh1, xxl + db.pin_offset_x[net_pin_id]);
                            yl1 = std::min(yl1, yyl + db.pin_offset_y[net_pin_id]);
                            yh1 = std::max(yh1, yyl + db.pin_offset_y[net_pin_id]);
                        }
                        
                        else {
                            xl0 = std::min(xl0, xxl);
                            xh0 = std::max(xh0, xxl);
                            yl0 = std::min(yl0, yyl);
                            yh0 = std::max(yh0, yyl);

                            xl1 = std::min(xl1, xxl);
                            xh1 = std::max(xh1, xxl);
                            yl1 = std::min(yl1, yyl);
                            yh1 = std::max(yh1, yyl);
                        }
                    } else {
                        permute_node_pin.push_back(net_pin_id);
                    }
                }
            }
            NetBoundingBox bound;
            bound.die[0] = Box(xl0, yl0 ,xh0, yh0);
            bound.die[1] = Box(xl1, yl1 ,xh1, yh1);
            netID2bound_map.insert({net_id, bound});
            netID2pinID_map.insert({net_id, permute_node_pin});
            
        }
    }
    
    // cells
    for (unsigned int k = 0; k < independent_set_size; ++k) {
        int node_id = independent_set[k];
        if (db.node_weight[node_id] == 0) continue;
        float node_width = db.node_size_x[node_id];
        bboxes[0].resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        bboxes[1].resize(db.flat_node2pin_start_map[node_id + 1] - db.flat_node2pin_start_map[node_id]);
        std::unordered_map<int, NetBoundingBox> netID2bound_map_update;
        for(auto kv : netID2pinID_map) {
            int net_id = kv.first;
            Box box0(db.xh, db.yh, db.xl, db.yl);
            Box box1(db.xh, db.yh, db.xl, db.yl);
            box0.xl = netID2bound_map.at(net_id).die[0].xl;
            box0.yl = netID2bound_map.at(net_id).die[0].yl;
            box0.xh = netID2bound_map.at(net_id).die[0].xh;
            box0.yh = netID2bound_map.at(net_id).die[0].yh;
            box1.xl = netID2bound_map.at(net_id).die[1].xl;
            box1.yl = netID2bound_map.at(net_id).die[1].yl;
            box1.xh = netID2bound_map.at(net_id).die[1].xh;
            box1.yh = netID2bound_map.at(net_id).die[1].yh;
            for(int i = 0; i < kv.second.size(); i++) {
                int net_pin_id = kv.second[i];
                Box box(db.xh, db.yh, db.xl, db.yl);
                int other_node_id = db.pin2node_map[net_pin_id];
                if (db.node_die[other_node_id] == -1) continue;
                if(other_node_id != node_id)  {
                    int c_id = db.node_die[other_node_id];
                    float xxl = db.x[other_node_id];
                    float yyl = db.y[other_node_id];
                    if (c_id == 0) {
                        box0.xl = std::min(box0.xl, xxl + db.pin_offset_x[net_pin_id]);
                        box0.xh = std::max(box0.xh, xxl + db.pin_offset_x[net_pin_id]);
                        box0.yl = std::min(box0.yl, yyl + db.pin_offset_y[net_pin_id]);
                        box0.yh = std::max(box0.yh, yyl + db.pin_offset_y[net_pin_id]);
                    } else if (c_id == 1) {
                        box1.xl = std::min(box1.xl, xxl + db.pin_offset_x[net_pin_id]);
                        box1.xh = std::max(box1.xh, xxl + db.pin_offset_x[net_pin_id]);
                        box1.yl = std::min(box1.yl, yyl + db.pin_offset_y[net_pin_id]);
                        box1.yh = std::max(box1.yh, yyl + db.pin_offset_y[net_pin_id]);
                    }
                    
                    else {
                        box0.xl = std::min(box0.xl, xxl);
                        box0.xh = std::max(box0.xh, xxl);
                        box0.yl = std::min(box0.yl, yyl);
                        box0.yh = std::max(box0.yh, yyl);

                        box1.xl = std::min(box1.xl, xxl);
                        box1.xh = std::max(box1.xh, xxl);
                        box1.yl = std::min(box1.yl, yyl);
                        box1.yh = std::max(box1.yh, yyl);
                    }
                }
                NetBoundingBox bound;
                bound.die[0] = box0;
                bound.die[1] = box1;
                netID2bound_map_update.insert({net_id, bound});
                
            }
        }
        int idx = 0;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id, ++idx) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];

            Box& box0 = bboxes[0][idx];
            Box& box1 = bboxes[1][idx];

            box0.xl = netID2bound_map_update.at(net_id).die[0].xl;
            box0.yl = netID2bound_map_update.at(net_id).die[0].yl;
            box0.xh = netID2bound_map_update.at(net_id).die[0].xh;
            box0.yh = netID2bound_map_update.at(net_id).die[0].yh;
            box1.xl = netID2bound_map_update.at(net_id).die[1].xl;
            box1.yl = netID2bound_map_update.at(net_id).die[1].yl;
            box1.xh = netID2bound_map_update.at(net_id).die[1].xh;
            box1.yh = netID2bound_map_update.at(net_id).die[1].yh;

            // cout << " ------1----- \n";
            // cout << box0.xl << endl;
            // cout << box1.xl << endl;
            // cout << box0.yl << endl;
            // cout << box1.yl << endl;
        }

        for (unsigned int j = 0; j < independent_set_size; ++j) {
            int pos_id = independent_set[j];
            if (db.node_weight[pos_id] == 0) continue;
            float target_x = db.x[pos_id];
            float target_y = db.y[pos_id];
            float target_hpwl = 0;

            auto const& target_space = state.spaces[pos_id];
            if (adjust_pos(target_x, node_width, target_space)) {
                // adjust_pos(target_x, db.node_size_x[pos_id], target_space);
                //  consider FENCE region

                idx = 0;
                for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                     node2pin_id < db.flat_node2pin_start_map[node_id + 1];
                     ++node2pin_id, ++idx) {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];

                    const Box& box0 = bboxes[0][idx];
                    const Box& box1 = bboxes[1][idx];

                    // cout << " ------2------ \n";
                    // cout << box0.xl << endl;
                    // cout << box1.xl << endl;
                    // cout << box0.yl << endl;
                    // cout << box1.yl << endl;
                    
                    if (db.net_mask[net_id]) {
                        float xxl = target_x;
                        float yyl = target_y;
                        float bxl0 = std::min(box0.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh0 = std::max(box0.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl0 = std::min(box0.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh0 = std::max(box0.yh, yyl + db.pin_offset_y[node_pin_id]);
                        float bxl1 = std::min(box1.xl, xxl + db.pin_offset_x[node_pin_id]);
                        float bxh1 = std::max(box1.xh, xxl + db.pin_offset_x[node_pin_id]);
                        float byl1 = std::min(box1.yl, yyl + db.pin_offset_y[node_pin_id]);
                        float byh1 = std::max(box1.yh, yyl + db.pin_offset_y[node_pin_id]);
                        target_hpwl += (bxh0 - bxl0) + (byh0 - byl0) + (bxh1 - bxl1) + (byh1 - byl1);


                    }

                    // cout << box0.xl << endl;
                    // cout << box1.xl << endl;
                    // cout << box0.yl << endl;
                    // cout << box1.yl << endl;

                    // cout << target_hpwl << endl;
                }
            } else {
                // dreamplaceAssertMsg(node_id != pos_id, "node %d, pos %d", node_id, pos_id);
                target_hpwl = state.large_number;
            }

            if (!major)  // row major
            {
                cost_matrix[independent_set.size() * k + j] = target_hpwl;
            } else  // column major
            {
                cost_matrix[independent_set.size() * j + k] = target_hpwl;
            }
        }
    }
}


void independentSetMatchingSequential(
    DetailedPlaceData& db, int num_bins_x, int num_bins_y, int set_size, int max_iters) {
    logger.info("============= Running DP: %d-ISM =============", max_iters);
    db.num_bins_x = num_bins_x;
    db.num_bins_y = num_bins_y;
    db.bin_size_x = (db.xh - db.xl) / num_bins_x;
    db.bin_size_y = (db.yh - db.yl) / num_bins_y;

    const double threshold = 0.00001 / 100;
    IndependentSetMatchingState state;
    state.batch_size = 1;
    state.set_size = set_size;
    state.num_moved = 0;
    state.large_number = (db.xh - db.xl + db.yh - db.yl) * 10;

    make_bin2node_map(db, db.x, db.y, db.node_size_x, db.node_size_y, state);
    construct_spaces(db, db.x, db.y, state.spaces, 1);

    state.grid_size = ceil_power2(std::max(db.num_bins_x, db.num_bins_y) / 8);
    state.max_diamond_search_sequence = state.grid_size * state.grid_size / 2;
    logger.info("diamond search grid size %d, sequence length %d", state.grid_size, state.max_diamond_search_sequence);

    state.ordered_nodes.resize(db.i_end);
    std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);
    state.independent_sets.resize(state.batch_size, std::vector<int>(state.set_size));
    state.dependent_markers.assign(db.num_nodes, 0);
    state.selected_markers.assign(db.i_end, 0);
    state.num_selected_markers.assign(db.i_end, 0);
    state.search_grids = diamond_search_sequence(state.grid_size, state.grid_size);
    // state.bin_marker.assign(db.num_bins_x*db.num_bins_y, 0);

    state.cost_matrices.resize(state.batch_size);
    state.solutions.resize(state.batch_size);
    state.orig_costs.resize(state.batch_size);
    state.target_costs.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    state.target_node2bin_map.resize(state.batch_size);
    state.target_spaces.resize(state.batch_size);
    AuctionAlgorithmCPULauncher<int> solver;
    bool major = false;  // row major

    int num_independent_sets = 0;

    std::vector<float> hpwls(max_iters + 1);
    hpwls[0] = db.compute_total_hpwl();
    logger.info("initial hpwl %g", hpwls[0]);
    for (int iter = 0; iter < max_iters; ++iter) {
        if (iter) {
            for (auto& bin2nodes : state.bin2node_map) {
                std::sort(bin2nodes.begin(), bin2nodes.end(), [&](int node_id1, int node_id2) {
                    return state.num_selected_markers[node_id1] < state.num_selected_markers[node_id2];
                });
                // std::random_shuffle(bin2nodes.begin(), bin2nodes.end());
            }
        }
        std::sort(state.ordered_nodes.begin(), state.ordered_nodes.end(), [&](int node_id1, int node_id2) {
            return state.num_selected_markers[node_id1] < state.num_selected_markers[node_id2];
        });
        // std::random_shuffle(state.ordered_nodes.begin(),
        // state.ordered_nodes.end());
        std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);
        // std::fill(state.bin_marker.begin(), state.bin_marker.end(), 0);

        for (int ii = 0; ii < db.i_end; ii += state.batch_size) {
            num_independent_sets = 0;
            for (int in_batch_id = 0; in_batch_id < state.batch_size; ++in_batch_id) {
                if (ii + in_batch_id < db.i_end) {
                    int node_id = state.ordered_nodes[ii + in_batch_id];
                    if (state.selected_markers[node_id]) {
                        continue;
                    }
                    if (db.node_weight[node_id] == 0) continue;
                    num_independent_sets += collect_independent_sets_sequential(db, state, node_id, in_batch_id);
                }
            }

#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < num_independent_sets; ++i) {
                auto const& independent_set = state.independent_sets.at(i);
                auto& cost_matrix = state.cost_matrices.at(i);
                cost_matrix.resize(independent_set.size() * independent_set.size());

                if (!db.via_dp) {
                    cost_matrix_construction2(db, state, major, i);
                } else {
                    cost_matrix_construction_via2(db, state, major, i);
                }
            }

#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < num_independent_sets; ++i) {
                auto const& independent_set = state.independent_sets.at(i);
                auto& cost_matrix = state.cost_matrices.at(i);
                auto& solution = state.solutions.at(i);
                auto& orig_cost = state.orig_costs.at(i);
                auto& target_cost = state.target_costs.at(i);
                solution.resize(independent_set.size());

                // solve bipartite assignment problem
                // compute initial cost
                orig_cost = 0;
                for (unsigned int j = 0; j < independent_set.size(); ++j) {
                    orig_cost += cost_matrix[j * independent_set.size() + j];
                }
                target_cost = solver.run(cost_matrix.data(), solution.data(), independent_set.size());
            }

#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < num_independent_sets; ++i) {
                apply_solution_sequential(db, state, i);
            }

            // if ((ii % ((int)ceil(db.i_end / 10.0))) == 0) {
            //     logger.debug("%d%%", (int(ii * 100 / db.i_end)));
            // }
        }

        hpwls[iter + 1] = db.compute_total_hpwl();
        logger.info(
            "iteration %d, target hpwl %g, delta %g(%g%%), solved %d "
            "sets, moved %g%% cells",
            iter,
            hpwls[iter + 1],
            hpwls[iter + 1] - hpwls[0],
            (hpwls[iter + 1] - hpwls[0]) / hpwls[0] * 100,
            num_independent_sets,
            state.num_moved / (double)db.i_end * 100);

        if (iter && hpwls[iter] - hpwls[iter + 1] < threshold * hpwls[iter]) {
            break;
        }
    }
}

}  // namespace dp