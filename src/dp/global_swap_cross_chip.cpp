
#include "global_swap_cross_chip.h"

namespace dpc {

void compute_search_bins(const DetailedPlaceData& db, SwapState& state, int begin, int end) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int node_id = begin; node_id < end; node_id += 1) {
        if (db.node_swap[node_id].item<int>() == 0) continue;
        // compute optimal region
        Box opt_box = (state.search_bin_strategy) ? db.compute_optimal_region(node_id)
                                                  : Box(db.x[node_id],
                                                        db.y[node_id],
                                                        db.x[node_id] + db.node_size_x[node_id],
                                                        db.y[node_id] + db.node_size_y[node_id]);
        int cx = db.pos2bin_x(opt_box.center_x());
        int cy = db.pos2bin_y(opt_box.center_y());
        state.search_bins[node_id] = cx * db.num_bins_y + cy;
    }
}  // END MODULE

//---------------------------------------------------------------------

// FIXME: space of which side
Space get_space(const DetailedPlaceData& db, const SwapState& state, int node_id) {
    auto const& row_id = state.node2row_map.at(node_id);
    auto const& c_id = row_id.chip_id;
    auto const& row2nodes = state.row2node_maps[c_id].at(row_id.row_id);
    Space space;
    space.xl = db.xl;
    space.xh = db.xh;
    if (row_id.sub_id) {
        int left_node_id = row2nodes[row_id.sub_id - 1];
        space.xl = std::max(space.xl, db.x[left_node_id] + db.node_size_x[left_node_id]);
    }
    if (row_id.sub_id + 1 < (int)row2nodes.size()) {
        int right_node_id = row2nodes[row_id.sub_id + 1];
        space.xh = std::min(space.xh, db.x[right_node_id]);
    }

    return db.align2site(space);
}  // END MODULE

//---------------------------------------------------------------------

void reset_state(DetailedPlaceData& db, SwapState& state) {
    state.candidates.resize(state.batch_size);
#pragma omp parallel for num_threads(state.num_threads)
    for (int i = 0; i < state.batch_size; ++i) {
        auto& candidates = state.candidates[i];
        candidates.clear();
        candidates.reserve(state.max_num_candidates);
    }

}  // END MODULE

//---------------------------------------------------------------------

float compute_positions_hint(const DetailedPlaceData& db,
                             const SwapState& state,
                             SwapCandidate& cand,
                             float node_xl,
                             float node_yl,
                             float node_width,
                             const Space& space) {
    // case I: two cells are horizontally abutting
    cand.node_xl[0][0] = node_xl;
    cand.node_yl[0][0] = node_yl;
    cand.node_xl[1][0] = db.x[cand.node_id[1]];
    cand.node_yl[1][0] = db.y[cand.node_id[1]];

    int c_id = db.node_weight[cand.node_id[0]];
    int other_c_id = db.node_weight[cand.node_id[1]];
    // assert(c_id != other_c_id); //FIXME:

    float target_node_width = db.node_size_xs[other_c_id][cand.node_id[1]];
    auto target_space = get_space(db, state, cand.node_id[1]);

    float node_width_sw = db.node_size_xs[other_c_id][cand.node_id[0]];
    float target_node_width_sw = db.node_size_xs[c_id][cand.node_id[1]];

    // FIXME: cross chip size & space

    if (space.xh < target_node_width_sw + space.xl || target_space.xh < node_width_sw + target_space.xl) {
        // some large number
        return std::numeric_limits<float>::max();
    }
    cand.node_xl[0][1] = cand.node_xl[1][0] + (target_node_width - node_width_sw) / 2;
    cand.node_xl[1][1] = cand.node_xl[0][0] + (node_width - target_node_width_sw) / 2;
    cand.node_xl[0][1] = db.align2site(cand.node_xl[0][1]);
    cand.node_xl[0][1] = std::max(cand.node_xl[0][1], target_space.xl);
    cand.node_xl[0][1] = std::min(cand.node_xl[0][1], target_space.xh - node_width_sw);
    cand.node_xl[1][1] = db.align2site(cand.node_xl[1][1]);
    cand.node_xl[1][1] = std::max(cand.node_xl[1][1], space.xl);
    cand.node_xl[1][1] = std::min(cand.node_xl[1][1], space.xh - target_node_width_sw);

    cand.node_yl[0][1] = cand.node_yl[1][0];
    cand.node_yl[1][1] = cand.node_yl[0][0];

    return 0;
}  // END MODULE

//---------------------------------------------------------------------

void collect_candidates(const DetailedPlaceData& db, SwapState& state, int idx_bgn, int idx_end) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int i = idx_bgn; i < idx_end; ++i) {
        int node_id = state.ordered_nodes.at(i);
        if (db.node_swap[node_id].item<int>() == 0) continue;
        int c_id = db.node_weight[node_id];

        float node_xl = db.x[node_id];
        float node_yl = db.y[node_id];
        float node_width = db.node_size_xs[c_id][node_id];
        auto space = get_space(db, state, node_id);
        int seed_bin_id = state.search_bins[node_id];
        int bx = seed_bin_id / db.num_bins_y;
        int by = seed_bin_id % db.num_bins_y;
        auto& candidates = state.candidates.at(i - idx_bgn);

        // FIXME: search opposite die / bin
        auto collect = [&](int ix, int iy, int c_id) {
            int bin_id = ix * db.num_bins_y + iy;
            auto const& bin2nodes = state.bin2node_maps[c_id].at(bin_id);
            int num_nodes_in_bin =
                state.bin2node_maps[c_id].at(bin_id).size() *
                (db.node_size_ys[c_id][node_id] == db.row_heights[c_id]);  // only consider single-row height
            int iters = std::min(state.max_num_candidates / 5, num_nodes_in_bin);

            for (int j = 0; j < iters; ++j) {
                if (db.node_swap[bin2nodes.at(j)].item<int>() == 0) continue;
                SwapCandidate cand;
                cand.node_id[0] = node_id;
                cand.node_id[1] = bin2nodes.at(j);
                if (db.node_size_ys[c_id][cand.node_id[1]] == db.row_heights[c_id]) {
                    cand.cost = compute_positions_hint(db, state, cand, node_xl, node_yl, node_width, space);
                    if (cand.cost == 0) {
                        candidates.push_back(cand);
                    }
                }
            }
        };

        // consider left, right, bottom, top bins
        collect(bx, by, 1 - c_id);
        if (bx) {
            collect(bx - 1, by, 1 - c_id);
        }
        if (bx + 1 < db.num_bins_x) {
            collect(bx + 1, by, 1 - c_id);
        }
        if (by) {
            collect(bx, by - 1, 1 - c_id);
        }
        if (by + 1 < db.num_bins_y) {
            collect(bx, by + 1, 1 - c_id);
        }

        // // consider this chip
        // collect(bx, by, c_id);
        // if (bx) {
        //     collect(bx - 1, by, c_id);
        // }
        // if (bx + 1 < db.num_bins_x) {
        //     collect(bx + 1, by, c_id);
        // }
        // if (by) {
        //     collect(bx, by - 1, c_id);
        // }
        // if (by + 1 < db.num_bins_y) {
        //     collect(bx, by + 1, c_id);
        // }
    }
}  // END MODULE

bool check_via_space(DetailedPlaceData& db, int node_id, float node_xl, float node_yl, int net_id) {
    Box box(db.xh, db.yh, db.xl, db.yl);
    const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];

    int net2pin_id = db.flat_net2pin_start_map[net_id];
    for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];
        if ((other_node_id < db.num_movable_nodes) ||
            ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
            int c_id = db.node_weight[other_node_id];
            float xxl = db.x[other_node_id];
            float yyl = db.y[other_node_id];

            // xxl+px
            xxl += db.pin_offset_xs[c_id][net_pin_id];
            // yyl+py
            yyl += db.pin_offset_ys[c_id][net_pin_id];

            box.xl = std::min(box.xl, xxl);
            box.xh = std::max(box.xh, xxl);
            box.yl = std::min(box.yl, yyl);
            box.yh = std::max(box.yh, yyl);
        }
    }
    int via_idx_xl = floor((box.xl - db.row_shift) / db.via_site_width);
    int via_idx_xh = floor((box.xh - db.row_shift) / db.via_site_width);
    int via_idx_yl = floor((box.yl - db.row_shift) / db.via_site_height);
    int via_idx_yh = floor((box.yh - db.row_shift) / db.via_site_height);

    assert(via_idx_xl <= via_idx_xh);
    assert(via_idx_yl <= via_idx_yh);
    for (int i = max(0, via_idx_xl); i <= via_idx_xh; i++) {
        for (int j = max(0, via_idx_yl); j <= via_idx_yh; j++) {
            if (db.via_map[i][j] == -1) {
                int64_t idx = net_id + db.num_movable_nodes;
                // cout << " adding " << idx << " at " << i << " " << j << endl;
                db.via_index[net_id][0] = i;
                db.via_index[net_id][1] = j;
                db.via_map[i][j] = idx;
                db.x[idx] = i * db.via_site_width + db.via_site_width / 2 + db.row_shift;
                db.y[idx] = j * db.via_site_height + db.via_site_height / 2 + db.row_shift;
                db.init_x[idx] = db.x[idx];
                db.init_y[idx] = db.y[idx];
                db.node_weight[idx] = 1 - db.node_weight[idx];
                return true;
            }
        }
    }

    return false;
}  // END MODULE

//---------------------------------------------------------------------

bool get_swap_net_info(
    DetailedPlaceData& db, int node_id, float node_xl, float node_yl, int target_node_id, vector<int>& switch_list) {
    auto node = db.nodes[node_id];
    int num_nets_node = node->Nets.size();
    int group = db.node_weight[node_id];
    vector<int> de_switch_list;
    vector<int> in_switch_list;

    for (unsigned n = 0; n != num_nets_node; ++n) {
        auto net = node->Nets[n];
        int num_nodes_net_node = net->Nodes.size();

        // TODO: which nets are cut
        int same_side = 0;
        int oppo_side = 0;
        int cut_flag = true;
        for (auto other_node : net->Nodes) {
            // TODO: other_node is connected to this node
            if (other_node->id == target_node_id) {
                cut_flag = false;
                break;
            }
            int other_group = db.node_weight[other_node->id];
            same_side += (other_group == group);
            oppo_side += (other_group != group);
        }
        /* switch the via on/off */
        if (cut_flag) {
            if (same_side == 1) {
                de_switch_list.push_back(net->id + db.num_movable_nodes);

            } else if (oppo_side == 0) {
                in_switch_list.push_back(net->id + db.num_movable_nodes);
            }
        }
    }

    /* check swap status: might be invalid */
    // if (de_switch_list.size() < in_switch_list.size()) return false; // FIXME:
    // assert(de_switch_list.size() >= in_switch_list.size()); // FIXME:
    for (int i = 0; i < de_switch_list.size(); i++) {
        int via_rm = de_switch_list[i];
        switch_list.push_back(via_rm);
        db.node_weight[via_rm] = 1 - db.node_weight[via_rm];

        // FIXME: via_map: remove vias first
        int x_id = db.via_index[via_rm - db.num_movable_nodes][0];
        int y_id = db.via_index[via_rm - db.num_movable_nodes][1];

        db.via_map[x_id][y_id] = -1;
    }

    /* update via node weight */
    for (int i = 0; i < in_switch_list.size(); i++) {
        int via_rm = de_switch_list[i];
        int via_ad = in_switch_list[i];

        bool has_space = check_via_space(db, node_id, node_xl, node_yl, via_ad - db.num_movable_nodes);
        if (!has_space) return false;
        switch_list.push_back(via_ad);

        // db.x[via_ad] = db.x[via_rm];
        // db.y[via_ad] = db.y[via_rm];
        // db.init_x[via_ad] = db.init_x[via_rm];
        // db.init_y[via_ad] = db.init_y[via_rm];
    }

    /* update cell node weight */

    // cout << " reverse " << node_id << " to " << 1 - db.node_weight[node_id] << endl;
    db.node_weight[node_id] = 1 - db.node_weight[node_id];
    switch_list.push_back(node_id);

    return true;
}

float compute_pair_hpwl_general(const DetailedPlaceData& db,
                                const SwapState& state,
                                int node_id,
                                float node_xl,
                                float node_yl,
                                int target_node_id,
                                float target_node_xl,
                                float target_node_yl,
                                // int c_id_prime,
                                int skip_node_id) {
    float cost = 0;
    int node2pin_id = db.flat_node2pin_start_map[node_id];
    const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
    // traverse nets connected to node_id
    for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        if (db.net_mask[net_id]) {
            vector<Box> boxs;
            boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);
            boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);
            const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
            for (int c_id = 0; c_id < 2; c_id++) {
                int net2pin_id = db.flat_net2pin_start_map[net_id];
                bool net_exist_flag = false;
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    // FIXME: allow invisible node
                    // FIXME: allow opposite chip
                    if (other_node_id == skip_node_id) {
                        net_exist_flag = false;
                        break;
                    }

                    // if ((((other_node_id != node_id) && (db.node_weight[other_node_id] == c_id) &&
                    //       (other_node_id < db.num_movable_nodes)) ||
                    //      ((db.node_weight[other_node_id] == 1) && (other_node_id >= db.num_movable_nodes)) ||
                    //      ((other_node_id == node_id) && (c_id == c_id_prime)))) {

                    if (((db.node_weight[other_node_id] == c_id) && (other_node_id < db.num_movable_nodes)) ||
                        ((db.node_weight[other_node_id] == 1) && (other_node_id >= db.num_movable_nodes))) {
                        // cout << "ADDING SIDE " << c_id << endl;
                        net_exist_flag = true;
                        float xxl;
                        float yyl;
                        if (other_node_id == node_id) {
                            xxl = node_xl;
                            yyl = node_yl;
                        } else if (other_node_id == target_node_id) {
                            xxl = target_node_xl;
                            yyl = target_node_yl;
                        } else {
                            xxl = db.x[other_node_id];
                            yyl = db.y[other_node_id];
                        }

                        // xxl+px
                        xxl += db.pin_offset_xs[c_id][net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_ys[c_id][net_pin_id];

                        // cout << c_id << " : " << xxl << " / " << yyl << " bbox of " << boxs[c_id].xh << " - "
                        //      << boxs[c_id].xl << " + " << boxs[c_id].yh << " - " << boxs[c_id].yl << endl;
                        boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                        boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                        boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                        boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);

                        // cout << c_id << " ------> " << boxs[c_id].xh << " - " << boxs[c_id].xl << " + " <<
                        // boxs[c_id].yh
                        //      << " - " << boxs[c_id].yl << endl;
                    }
                }
                if (!net_exist_flag) {
                    boxs[c_id].xl = 0;
                    boxs[c_id].xh = 0;
                    boxs[c_id].yl = 0;
                    boxs[c_id].yh = 0;
                }
                cost += (boxs[c_id].xh - boxs[c_id].xl + boxs[c_id].yh - boxs[c_id].yl);

                // cout << "Net " << net_id << " at chip " << c_id << "  "
                //      << (boxs[c_id].xh - boxs[c_id].xl + boxs[c_id].yh - boxs[c_id].yl) << endl;
            }
        }
    }

    // cout << " \n";

    return cost;
}  // END MODULE

//---------------------------------------------------------------------

void recover_swap(DetailedPlaceData& db, vector<int>& switch_list) {
    for (int node_id : switch_list) {
        // cout << " recover " << node_id << " to " << 1 - db.node_weight[node_id] << endl;
        db.node_weight[node_id] = 1 - db.node_weight[node_id];
        if (node_id >= db.num_movable_nodes) {
            // FIXME: via_map: remove vias first
            int x_id = db.via_index[node_id - db.num_movable_nodes][0];
            int y_id = db.via_index[node_id - db.num_movable_nodes][1];

            int idx = db.via_map[x_id][y_id];
            if (idx == node_id) {  // via_map[][] = via_id remove
                db.via_map[x_id][y_id] = -1;
            } else {
                db.via_map[x_id][y_id] = node_id;
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

// FIXME: hpwl
void compute_candidate_cost(DetailedPlaceData& db, SwapState& state) {
    for (int i = 0; i < state.batch_size; i += 1) {
        auto& candidates = state.candidates.at(i);
        for (unsigned int j = 0; j < candidates.size(); ++j) {
            auto& cand = candidates[j];

            if (cand.node_id[0] < db.num_movable_nodes && cand.node_id[1] < db.num_movable_nodes) {
                cand.cost = -compute_pair_hpwl_general(db,
                                                       state,
                                                       cand.node_id[0],
                                                       cand.node_xl[0][0],
                                                       cand.node_yl[0][0],
                                                       cand.node_id[1],
                                                       cand.node_xl[1][0],
                                                       cand.node_yl[1][0],
                                                       std::numeric_limits<int>::max());
                cand.cost -= compute_pair_hpwl_general(db,
                                                       state,
                                                       cand.node_id[1],
                                                       cand.node_xl[1][0],
                                                       cand.node_yl[1][0],
                                                       cand.node_id[0],
                                                       cand.node_xl[0][0],
                                                       cand.node_yl[0][0],
                                                       cand.node_id[0]);

                vector<int> switch_list1;
                vector<int> switch_list2;
                bool swap_status1 = get_swap_net_info(
                    db, cand.node_id[0], cand.node_xl[0][1], cand.node_yl[0][1], cand.node_id[1], switch_list1);
                bool swap_status2 = get_swap_net_info(
                    db, cand.node_id[1], cand.node_xl[1][1], cand.node_yl[1][1], cand.node_id[0], switch_list2);
                if (!swap_status1 || !swap_status2) {
                    recover_swap(db, switch_list1);
                    recover_swap(db, switch_list2);
                    cand.cost = 0;
                    continue;
                }

                cand.switch_list1 = switch_list1;
                cand.switch_list2 = switch_list2;
                cand.cost += compute_pair_hpwl_general(db,
                                                       state,
                                                       cand.node_id[0],
                                                       cand.node_xl[0][1],
                                                       cand.node_yl[0][1],
                                                       cand.node_id[1],
                                                       cand.node_xl[1][1],
                                                       cand.node_yl[1][1],
                                                       std::numeric_limits<int>::max());
                cand.cost += compute_pair_hpwl_general(db,
                                                       state,
                                                       cand.node_id[1],
                                                       cand.node_xl[1][1],
                                                       cand.node_yl[1][1],
                                                       cand.node_id[0],
                                                       cand.node_xl[0][1],
                                                       cand.node_yl[0][1],
                                                       cand.node_id[0]);

                /* reverse swapped vias */
                // FIXME:

                recover_swap(db, switch_list1);
                recover_swap(db, switch_list2);
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void reduce_min_2d(const SwapState& state, std::vector<std::vector<SwapCandidate> >& candidates, int batch_size) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int i = 0; i < batch_size; ++i) {
        auto& row_candidates = candidates.at(i);
        for (unsigned int j = 1; j < row_candidates.size(); ++j) {
            if (row_candidates[j].cost < row_candidates[0].cost) {
                row_candidates[0] = row_candidates[j];
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void mark_dependent_nodes(const DetailedPlaceDBType& db,
                          IndependentSetMatchingStateType& state,
                          int node_id,
                          unsigned char value) {
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
                if (other_node_id < db.num_nodes)  // other_node_id may exceed db.num_nodes like IO pins
                {
                    state.node_markers[other_node_id] = value;
                }
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void apply_candidates(DetailedPlaceData& db, SwapState& state, int num_candidates, int& count) {
    for (int i = 0; i < num_candidates; ++i) {
        auto const& row_candidates = state.candidates.at(i);
        if (row_candidates.empty()) {
            continue;
        }
        auto const& best_cand = row_candidates.at(0);

        if (best_cand.cost < 0) {
        // if (true) {
            // if (best_cand.cost < 0 &&
            //     !(state.node_markers.at(best_cand.node_id[0]) || state.node_markers.at(best_cand.node_id[1]))) {
            float node_width = db.node_size_x[best_cand.node_id[0]];
            float target_node_width = db.node_size_x[best_cand.node_id[1]];
            Space space = get_space(db, state, best_cand.node_id[0]);
            Space target_space = get_space(db, state, best_cand.node_id[1]);

            int c_id = db.node_weight[best_cand.node_id[0]];
            int other_c_id = db.node_weight[best_cand.node_id[1]];
            assert(c_id != other_c_id);

            float node_width_sw = db.node_size_xs[other_c_id][best_cand.node_id[0]];
            float target_node_width_sw = db.node_size_xs[c_id][best_cand.node_id[1]];

            // space may no longer be large enough or the previously computed
            // locations may not be correct any more
            if (best_cand.node_xl[0][1] >= target_space.xl &&
                best_cand.node_xl[0][1] + node_width_sw <= target_space.xh && best_cand.node_xl[1][1] >= space.xl &&
                best_cand.node_xl[1][1] + target_node_width_sw <= space.xh) {
                // cout << " ---------------------- swapping with -------------------------- " << best_cand.cost <<
                // endl; cout << best_cand.node_id[0] << " and " << best_cand.node_id[1] << endl;

                /* update vias */
                vector<int> switch_list1;
                vector<int> switch_list2;
                bool swap_status1 = get_swap_net_info(db,
                                                      best_cand.node_id[0],
                                                      best_cand.node_xl[0][1],
                                                      best_cand.node_yl[0][1],
                                                      best_cand.node_id[1],
                                                      switch_list1);
                bool swap_status2 = get_swap_net_info(db,
                                                      best_cand.node_id[1],
                                                      best_cand.node_xl[1][1],
                                                      best_cand.node_yl[1][1],
                                                      best_cand.node_id[0],
                                                      switch_list2);
                if (!swap_status1 || !swap_status2) {
                    recover_swap(db, switch_list1);
                    recover_swap(db, switch_list2);
                    continue;
                }

                // state.node_markers[best_cand.node_id[0]] = 1;
                // state.node_markers[best_cand.node_id[1]] = 1;
                mark_dependent_nodes(db, state, best_cand.node_id[0], 1);
                mark_dependent_nodes(db, state, best_cand.node_id[1], 1);

                BinMapIndex& bin_id = state.node2bin_map.at(best_cand.node_id[0]);
                BinMapIndex& target_bin_id = state.node2bin_map.at(best_cand.node_id[1]);
                RowMapIndex& row_id = state.node2row_map.at(best_cand.node_id[0]);
                RowMapIndex& target_row_id = state.node2row_map.at(best_cand.node_id[1]);

                auto& row2nodes = state.row2node_maps[c_id].at(row_id.row_id);
                auto& target_row2nodes = state.row2node_maps[other_c_id].at(target_row_id.row_id);

                // logger.info("swap node %d from (%.2f, %.2f) to (%.2f, %.2f) / %d to %d",
                //             best_cand.node_id[0],
                //             best_cand.node_xl[0][0],
                //             best_cand.node_yl[0][0],
                //             best_cand.node_xl[0][1],
                //             best_cand.node_yl[0][1],
                //             c_id,
                //             db.node_weight[best_cand.node_id[0]]);
                // logger.info("swap node %d from (%.2f, %.2f) to (%.2f, %.2f) / %d to %d",
                //             best_cand.node_id[1],
                //             best_cand.node_xl[1][0],
                //             best_cand.node_yl[1][0],
                //             best_cand.node_xl[1][1],
                //             best_cand.node_yl[1][1],
                //             other_c_id,
                //             db.node_weight[best_cand.node_id[1]]);
                // logger.info("space %.2f, %.2f", space.xl, target_space.xl);
                // logger.info("space %.2f, %.2f", space.xh, target_space.xh);

                db.x[best_cand.node_id[0]] = best_cand.node_xl[0][1];
                db.y[best_cand.node_id[0]] = best_cand.node_yl[0][1];
                db.x[best_cand.node_id[1]] = best_cand.node_xl[1][1];
                db.y[best_cand.node_id[1]] = best_cand.node_yl[1][1];
                int& bin2node_map_node_id = state.bin2node_maps[c_id].at(bin_id.bin_id).at(bin_id.sub_id);
                int& bin2node_map_target_node_id =
                    state.bin2node_maps[other_c_id].at(target_bin_id.bin_id).at(target_bin_id.sub_id);
                std::swap(bin2node_map_node_id, bin2node_map_target_node_id);
                std::swap(bin_id, target_bin_id);

                // update row2node_map and node2row_map
                std::swap(row2nodes[row_id.sub_id], target_row2nodes[target_row_id.sub_id]);
                std::swap(row_id, target_row_id);

                /* update node size */
                db.node_size_x[best_cand.node_id[0]] = db.node_size_xs[1 - c_id][best_cand.node_id[0]];
                db.node_size_x[best_cand.node_id[1]] = db.node_size_xs[1 - other_c_id][best_cand.node_id[1]];
                db.node_size_y[best_cand.node_id[0]] = db.node_size_ys[1 - c_id][best_cand.node_id[0]];
                db.node_size_y[best_cand.node_id[1]] = db.node_size_ys[1 - other_c_id][best_cand.node_id[1]];

                db.init_x[best_cand.node_id[0]] = db.x[best_cand.node_id[0]] + db.node_size_x[best_cand.node_id[0]] / 2;
                db.init_x[best_cand.node_id[1]] = db.x[best_cand.node_id[1]] + db.node_size_x[best_cand.node_id[1]] / 2;
                db.init_y[best_cand.node_id[0]] = db.y[best_cand.node_id[0]] + db.node_size_y[best_cand.node_id[0]] / 2;
                db.init_y[best_cand.node_id[1]] = db.y[best_cand.node_id[1]] + db.node_size_y[best_cand.node_id[1]] / 2;

                db.a = best_cand.node_id[0];
                db.b = best_cand.node_id[1];

                // space = get_space(db, state, best_cand.node_id[0]);
                // target_space = get_space(db, state, best_cand.node_id[1]);
                // logger.info("space %.2f, %.2f", space.xl, target_space.xl);
                // logger.info("space %.2f, %.2f", space.xh, target_space.xh);

                count++;
            }
        }
    }

    for (int i = 0; i < num_candidates; ++i) {
        auto const& row_candidates = state.candidates.at(i);
        if (row_candidates.empty()) {
            continue;
        }
        auto const& best_cand = row_candidates.at(0);
        mark_dependent_nodes(db, state, best_cand.node_id[0], 0);
        mark_dependent_nodes(db, state, best_cand.node_id[1], 0);
    }
}  // END MODULE

//---------------------------------------------------------------------

void global_swap(DetailedPlaceData& db, SwapState& state) {
    int collect_candidates_runs = 0, compute_candidate_cost_runs = 0, reduce_min_2d_runs = 0, apply_candidates_runs = 0;

    compute_search_bins(db, state, 0, db.num_movable_nodes);

    int count = 0;
    for (int i = 0; i < db.num_movable_nodes; i += state.batch_size) {
        // all results are stored in state.candidates
        int idx_bgn = i;
        int idx_end = std::min(i + state.batch_size, db.num_movable_nodes);

        reset_state(db, state);

        collect_candidates(db, state, idx_bgn, idx_end);
        collect_candidates_runs += 1;

        compute_candidate_cost(db, state);
        compute_candidate_cost_runs += 1;

        // check_candidate_costs(db, state);
        // reduce min and apply
        reduce_min_2d(state, state.candidates, state.batch_size);
        reduce_min_2d_runs += 1;

        // check_candidate_costs(db, state);
        // must use single thread
        apply_candidates(db, state, idx_end - idx_bgn, count);
        apply_candidates_runs += 1;

        // if (count > 0) break;  // FIXME:
    }
}  // END MODULE

//---------------------------------------------------------------------

void globalSwapCrossChip(DetailedPlaceData& db,
                         const torch::Tensor node_swap,  // TODO: whether a node is swappable
                         torch::Tensor numRows,
                         torch::Tensor rowHeights,
                         int num_bins_x,
                         int num_bins_y,
                         int max_iters,
                         int batch_size) {
    logger.info("============= Running DP: Global Swap Cross-Chip=============");
    db.num_bins_x = num_bins_x;
    db.num_bins_y = num_bins_y;
    db.bin_size_x = (db.xh - db.xl) / num_bins_x;
    db.bin_size_y = (db.yh - db.yl) / num_bins_y;
    if (node_swap.numel()) {
        db.node_swap = node_swap;
    }

    SwapState state;
    state.num_threads = db.num_threads;

    const float stop_threshold = 0.1 / 100;
    state.batch_size = batch_size;  // TODO: #visible nodes
    int max_num_candidates_per_row = (2 << (int)log2(ceil(sqrt(db.num_nodes / (db.num_bins_x * db.num_bins_y)))));
    state.max_num_candidates =
        (1 << (int)ceil(log2(ceil(db.bin_size_y / db.row_height)))) * max_num_candidates_per_row * 5 * 2;  // TODO:
    state.max_num_candidates_all = state.batch_size * state.max_num_candidates;
    state.search_bin_strategy = 1;

    logger.info("#binX: %d, #binY: %d", num_bins_x, num_bins_y);
    logger.info("#Candidates: /row: %d, max: %d, all: %d",
                max_num_candidates_per_row,
                state.max_num_candidates,
                state.max_num_candidates_all);

    // distribute cells to rows
    state.row2node_maps.resize(2);
    for (int c_id = 0; c_id < 2; c_id++) {
        db.num_sites_ys.push_back(numRows[c_id].item<int>());
        db.row_heights.push_back(rowHeights[c_id].item<float>());
        state.row2node_maps[c_id].resize(db.num_sites_ys[c_id]);
    }
    state.node2row_map.resize(db.num_nodes);
    db.make_row2node_map(db.x, db.y, state.row2node_maps);
    for (int c_id = 0; c_id < 2; c_id++) {
        for (int i = 0; i < db.num_sites_ys[c_id]; ++i) {
            for (unsigned int j = 0; j < state.row2node_maps[c_id][i].size(); ++j) {
                auto& row_id = state.node2row_map.at(state.row2node_maps[c_id][i][j]);
                row_id.row_id = i;
                row_id.sub_id = j;
                row_id.chip_id = c_id;
            }
        }
    }

    // TODO:
    // distribute cells to bin
    state.bin2node_maps.resize(2);
    for (int c_id = 0; c_id < 2; c_id++) {
        state.bin2node_maps[c_id].resize(db.num_bins_x * db.num_bins_y);
    }
    state.node2bin_map.resize(db.num_movable_nodes);

    db.make_bin2node_map(db.x, db.y, db.node_size_x, db.node_size_y, state.bin2node_maps, state.node2bin_map);


    state.ordered_nodes.resize(db.num_movable_nodes);
    std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);

    state.candidates.resize(state.batch_size);
    state.search_bins.resize(db.num_movable_nodes);
    state.net_hpwls.resize(db.num_nets);
    state.node_markers.assign(db.num_nodes, 0);

    std::vector<float> hpwls(max_iters + 1);
    hpwls.at(0) = db.compute_total_hpwl_concurrent_c(db.x, db.y, state.net_hpwls.data());
    logger.info("initial Chpwl = %.3f", hpwls[0]);

    for (int iter = 0; iter < max_iters; ++iter) {
        // for (int iter = 0; iter < 1; ++iter) {
        std::random_shuffle(state.ordered_nodes.begin(), state.ordered_nodes.end());
        global_swap(db, state);

        // FIXME: update pin rel pos hpwl
        for (int64_t i = 0; i < db.num_pins; i++) {
            int64_t node_id = db.pin_id2node_id[i].item<int64_t>();
            int c_id = db.node_weight[node_id];
            db.pin_offset_x[i] = db.pin_offset_xs[c_id][i];
            db.pin_offset_y[i] = db.pin_offset_ys[c_id][i];
            db.pin_rel_cpos[i] = (c_id ? db.pin_rel_cpos_top[i] : db.pin_rel_cpos_bot[i]);
        }

        hpwls[iter + 1] = db.compute_total_hpwl_concurrent_c(db.x, db.y, state.net_hpwls.data());
        logger.info("iteration %d: Chpwl %.3f => %.3f (imp. %g%%)",
                    iter,
                    hpwls[0],
                    hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);
        state.search_bin_strategy = !state.search_bin_strategy;

        // if ((iter & 1) && hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
        //     break;
        // }
    }
    for (int64_t i = db.num_movable_nodes; i < db.num_nodes; i++) {
        if (db.node_weight[i] == 1) {
            db.node_size_x[i] = db.via_site_width;
            db.node_size_y[i] = db.via_site_height;
            db.node_weight[i] = 2;
        } else {
            db.node_size_x[i] = 0;
            db.node_size_y[i] = 0;
            db.node_weight[i] = -1;
        }
    }
}  // END MODULE

}  // namespace dpc