
#include "global_swap.h"

namespace dp {

void compute_search_bins(const DetailedPlaceData& db, SwapState& state, int begin, int end) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int node_id = begin; node_id < end; node_id += 1) {
        if (db.node_weight[node_id] == 0) continue;
        // compute optimal region
        Box opt_box = (state.search_bin_strategy) ? db.compute_optimal_region(node_id)
                                                  : Box(db.x[node_id],
                                                        db.y[node_id],
                                                        db.x[node_id] + db.node_size_x[node_id],
                                                        db.y[node_id] + db.node_size_y[node_id]);
        // Box<T> opt_box = Box<T>(db.x[node_id],
        //        db.y[node_id],
        //        db.x[node_id]+db.node_size_x[node_id],
        //        db.y[node_id]+db.node_size_y[node_id]);
        int cx = db.pos2bin_x(opt_box.center_x());
        int cy = db.pos2bin_y(opt_box.center_y());
        state.search_bins[node_id] = cx * db.num_bins_y + cy;
    }
}  // END MODULE

//---------------------------------------------------------------------

Space get_space(const DetailedPlaceData& db, const SwapState& state, int node_id) {
    auto const& row_id = state.node2row_map.at(node_id);
    auto const& row2nodes = state.row2node_map.at(row_id.row_id);
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
    // align space to sites
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
    float target_node_width = db.node_size_x[cand.node_id[1]];
    auto target_space = get_space(db, state, cand.node_id[1]);
    if (space.xh >= target_space.xl && target_space.xh >= space.xl &&
        cand.node_yl[0][0] == cand.node_yl[1][0])  // case I: abutting, not
                                                   // exactly abutting, there
                                                   // might be space between two
                                                   // cells, this is a generalized
                                                   // case
    {
        if (cand.node_xl[0][0] < cand.node_xl[1][0]) {
            cand.node_xl[0][1] = cand.node_xl[1][0] + target_node_width - node_width;
            cand.node_xl[1][1] = cand.node_xl[0][0];
        } else {
            cand.node_xl[0][1] = cand.node_xl[1][0];
            cand.node_xl[1][1] = cand.node_xl[0][0] + node_width - target_node_width;
        }
    } else  // case II: not abutting
    {
        if (space.xh < target_node_width + space.xl || target_space.xh < node_width + target_space.xl) {
            // some large number
            return std::numeric_limits<float>::max();
        }
        cand.node_xl[0][1] = cand.node_xl[1][0] + (target_node_width - node_width) / 2;
        cand.node_xl[1][1] = cand.node_xl[0][0] + (node_width - target_node_width) / 2;
        cand.node_xl[0][1] = db.align2site(cand.node_xl[0][1]);
        cand.node_xl[0][1] = std::max(cand.node_xl[0][1], target_space.xl);
        cand.node_xl[0][1] = std::min(cand.node_xl[0][1], target_space.xh - node_width);
        cand.node_xl[1][1] = db.align2site(cand.node_xl[1][1]);
        cand.node_xl[1][1] = std::max(cand.node_xl[1][1], space.xl);
        cand.node_xl[1][1] = std::min(cand.node_xl[1][1], space.xh - target_node_width);
    }
    cand.node_yl[0][1] = cand.node_yl[1][0];
    cand.node_yl[1][1] = cand.node_yl[0][0];

    return 0;
}  // END MODULE

//---------------------------------------------------------------------

void collect_candidates(const DetailedPlaceData& db, SwapState& state, int idx_bgn, int idx_end) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int i = idx_bgn; i < idx_end; ++i) {
        int node_id = state.ordered_nodes.at(i);
        if (db.node_weight[node_id] == 0) continue;
        float node_xl = db.x[node_id];
        float node_yl = db.y[node_id];
        float node_width = db.node_size_x[node_id];
        auto space = get_space(db, state, node_id);
        int seed_bin_id = state.search_bins[node_id];
        int bx = seed_bin_id / db.num_bins_y;
        int by = seed_bin_id % db.num_bins_y;
        auto& candidates = state.candidates.at(i - idx_bgn);

        auto collect = [&](int ix, int iy) {
            int bin_id = ix * db.num_bins_y + iy;
            auto const& bin2nodes = state.bin2node_map.at(bin_id);
            int num_nodes_in_bin = state.bin2node_map.at(bin_id).size() *
                                   (db.node_size_y[node_id] == db.row_height);  // only consider single-row height cell
            int iters = std::min(state.max_num_candidates / 5, num_nodes_in_bin);

            for (int j = 0; j < iters; ++j) {
                SwapCandidate cand;
                cand.node_id[0] = node_id;
                cand.node_id[1] = bin2nodes.at(j);
                if (db.node_size_y[cand.node_id[1]] == db.row_height) {
                    cand.cost = compute_positions_hint(db, state, cand, node_xl, node_yl, node_width, space);
                    if (cand.cost == 0) {
                        candidates.push_back(cand);
                    }
                }
            }
        };

        // consider left, right, bottom, top bins
        collect(bx, by);
        if (bx) {
            collect(bx - 1, by);
        }
        if (bx + 1 < db.num_bins_x) {
            collect(bx + 1, by);
        }
        if (by) {
            collect(bx, by - 1);
        }
        if (by + 1 < db.num_bins_y) {
            collect(bx, by + 1);
        }
    }
}  // END MODULE

//---------------------------------------------------------------------
void compute_bound_map(const DetailedPlaceData& db,
                        const SwapState& state,
                        int node_id,
                        int target_node_id,
                        int skip_node_id, 
                        std::unordered_map<int, NetBoundingBox> &netID2bound_map,
                        std::unordered_map<int, vector<int>> &netID2pinID_map) {
    int node2pin_id = db.flat_node2pin_start_map[node_id];
    const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
    if (!db.via_dp) {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                vector<int> permute_node_pin;
                Box box(db.xh, db.yh, db.xl, db.yl);
                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if ((other_node_id != skip_node_id) && (db.node_die[other_node_id] == 1)) {
                        float xxl;
                        float yyl;
                        if ((other_node_id == node_id) || (other_node_id == target_node_id)) {
                            permute_node_pin.push_back(net_pin_id);
                            continue;
                        } else {
                            xxl = db.x[other_node_id];
                            yyl = db.y[other_node_id];
                        }
                        // xxl+px
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        box.xl = std::min(box.xl, xxl);
                        box.xh = std::max(box.xh, xxl);
                        box.yl = std::min(box.yl, yyl);
                        box.yh = std::max(box.yh, yyl);
                    }
                }
                NetBoundingBox bound;
                bound.die[0] = box;
                netID2bound_map.insert({net_id, bound});
                netID2pinID_map.insert({net_id, permute_node_pin});
            }
        }
    } else {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                vector<int> permute_node_pin;
                vector<Box> boxs;
                boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);
                boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);

                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if ((other_node_id < db.num_movable_nodes) ||
                        ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
                        float xxl;
                        float yyl;
                        if ((other_node_id == node_id) || (other_node_id == target_node_id)) {
                            permute_node_pin.push_back(net_pin_id);
                            continue;
                        }else {
                            xxl = db.x[other_node_id];
                            yyl = db.y[other_node_id];
                        }

                        int c_id = db.node_die[other_node_id];
                        // xxl+px
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        if (c_id != 2) {
                            boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                            boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                            boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                            boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                        } else {
                            for (c_id = 0; c_id < 2; c_id++) {
                                boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                                boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                                boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                                boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                            }
                        }
                    }
                }
                NetBoundingBox bound;
                bound.die[0] = boxs[0];
                bound.die[1] = boxs[1];
                netID2bound_map.insert({net_id, bound});
                netID2pinID_map.insert({net_id, permute_node_pin});

            }
        }
    }
    
}

float compute_pair_hpwl_general_with_precompute(const DetailedPlaceData& db,
                                const SwapState& state,
                                int node_id,
                                float node_xl,
                                float node_yl,
                                int target_node_id,
                                float target_node_xl,
                                float target_node_yl,
                                int skip_node_id,
                                std::unordered_map<int, NetBoundingBox> &netID2bound_map,
                                std::unordered_map<int, vector<int>> &netID2pinID_map) {
    float cost = 0;
    int node2pin_id = db.flat_node2pin_start_map[node_id];
    const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
    if (!db.via_dp) {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                Box box(db.xh, db.yh, db.xl, db.yl);
                box.xl = netID2bound_map.at(net_id).die[0].xl;
                box.yl = netID2bound_map.at(net_id).die[0].yl;
                box.xh = netID2bound_map.at(net_id).die[0].xh;
                box.yh = netID2bound_map.at(net_id).die[0].yh;
                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if(!((other_node_id == node_id) || (other_node_id == target_node_id))) continue;
                    if ((other_node_id != skip_node_id) && (db.node_die[other_node_id] == 1)) {
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
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        box.xl = std::min(box.xl, xxl);
                        box.xh = std::max(box.xh, xxl);
                        box.yl = std::min(box.yl, yyl);
                        box.yh = std::max(box.yh, yyl);
                    }
                }
                cost += (box.xh - box.xl + box.yh - box.yl);
            }
        }
    } else {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                vector<Box> boxs;
                boxs.emplace_back(netID2bound_map.at(net_id).die[0].xl, netID2bound_map.at(net_id).die[0].yl, netID2bound_map.at(net_id).die[0].xh, netID2bound_map.at(net_id).die[0].yh);
                boxs.emplace_back(netID2bound_map.at(net_id).die[1].xl, netID2bound_map.at(net_id).die[1].yl, netID2bound_map.at(net_id).die[1].xh, netID2bound_map.at(net_id).die[1].yh);


                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if(!((other_node_id == node_id) || (other_node_id == target_node_id))) continue;
                    if ((other_node_id < db.num_movable_nodes) ||
                        ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
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

                        int c_id = db.node_die[other_node_id];
                        // xxl+px
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        if (c_id != 2) {
                            boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                            boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                            boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                            boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                        } else {
                            for (c_id = 0; c_id < 2; c_id++) {
                                boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                                boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                                boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                                boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                            }
                        }
                    }
                }
                cost += (boxs[0].xh - boxs[0].xl + boxs[0].yh - boxs[0].yl) +
                        (boxs[1].xh - boxs[1].xl + boxs[1].yh - boxs[1].yl);
            }
        }
    }
    return cost;
}


float compute_pair_hpwl_general(const DetailedPlaceData& db,
                                const SwapState& state,
                                int node_id,
                                float node_xl,
                                float node_yl,
                                int target_node_id,
                                float target_node_xl,
                                float target_node_yl,
                                int skip_node_id
                                ) {
    float cost = 0;
    int node2pin_id = db.flat_node2pin_start_map[node_id];
    const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
    if (!db.via_dp) {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                Box box(db.xh, db.yh, db.xl, db.yl);
                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if ((other_node_id != skip_node_id) && (db.node_die[other_node_id] == 1)) {
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
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        box.xl = std::min(box.xl, xxl);
                        box.xh = std::max(box.xh, xxl);
                        box.yl = std::min(box.yl, yyl);
                        box.yh = std::max(box.yh, yyl);
                    }
                }
                cost += (box.xh - box.xl + box.yh - box.yl);
            }
        }
    } else {
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id]) {
                vector<Box> boxs;
                boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);
                boxs.emplace_back(db.xh, db.yh, db.xl, db.yl);

                int net2pin_id = db.flat_net2pin_start_map[net_id];
                const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
                for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if ((other_node_id < db.num_movable_nodes) ||
                        ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
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

                        int c_id = db.node_die[other_node_id];
                        // xxl+px
                        xxl += db.pin_offset_x[net_pin_id];
                        // yyl+py
                        yyl += db.pin_offset_y[net_pin_id];
                        if (c_id != 2) {
                            boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                            boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                            boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                            boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                        } else {
                            for (c_id = 0; c_id < 2; c_id++) {
                                boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                                boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                                boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                                boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                            }
                        }
                    }
                }
                cost += (boxs[0].xh - boxs[0].xl + boxs[0].yh - boxs[0].yl) +
                        (boxs[1].xh - boxs[1].xl + boxs[1].yh - boxs[1].yl);
            }
        }
    }
    return cost;
}  // END MODULE

//---------------------------------------------------------------------

void compute_candidate_cost(const DetailedPlaceData& db, SwapState& state) {
#pragma omp parallel for num_threads(state.num_threads)
    for (int i = 0; i < state.batch_size; i += 1) {
        auto& candidates = state.candidates.at(i);
        for (unsigned int j = 0; j < candidates.size(); ++j) {
            auto& cand = candidates[j];
            if (cand.node_id[0] < db.i_end && cand.node_id[1] < db.i_end) {
                // TODO: consider fence
                std::unordered_map<int, NetBoundingBox> netID2bound_map;
                std::unordered_map<int, vector<int>> netID2pinID_map;
                compute_bound_map(db, state, cand.node_id[0], cand.node_id[1], std::numeric_limits<int>::max(), netID2bound_map, netID2pinID_map);
                compute_bound_map(db, state, cand.node_id[1], cand.node_id[0], std::numeric_limits<int>::max(), netID2bound_map, netID2pinID_map);


                cand.cost = -compute_pair_hpwl_general_with_precompute(db,
                                                       state,
                                                       cand.node_id[0],
                                                       cand.node_xl[0][0],
                                                       cand.node_yl[0][0],
                                                       cand.node_id[1],
                                                       cand.node_xl[1][0],
                                                       cand.node_yl[1][0],
                                                       std::numeric_limits<int>::max(),
                                                       netID2bound_map, netID2pinID_map);
                cand.cost -= compute_pair_hpwl_general_with_precompute(db,
                                                       state,
                                                       cand.node_id[1],
                                                       cand.node_xl[1][0],
                                                       cand.node_yl[1][0],
                                                       cand.node_id[0],
                                                       cand.node_xl[0][0],
                                                       cand.node_yl[0][0],
                                                       cand.node_id[0],
                                                       netID2bound_map, netID2pinID_map);
                cand.cost += compute_pair_hpwl_general_with_precompute(db,
                                                       state,
                                                       cand.node_id[0],
                                                       cand.node_xl[0][1],
                                                       cand.node_yl[0][1],
                                                       cand.node_id[1],
                                                       cand.node_xl[1][1],
                                                       cand.node_yl[1][1],
                                                       std::numeric_limits<int>::max(),
                                                       netID2bound_map, netID2pinID_map);
                cand.cost += compute_pair_hpwl_general_with_precompute(db,
                                                       state,
                                                       cand.node_id[1],
                                                       cand.node_xl[1][1],
                                                       cand.node_yl[1][1],
                                                       cand.node_id[0],
                                                       cand.node_xl[0][1],
                                                       cand.node_yl[0][1],
                                                       cand.node_id[0],
                                                       netID2bound_map, netID2pinID_map);
/*
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
*/
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
                if (other_node_id < db.num_movable_nodes)  // other_node_id may exceed db.num_nodes like IO pins
                {
                    state.node_markers[other_node_id] = value;
                }
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void apply_candidates(DetailedPlaceData& db, SwapState& state, int num_candidates) {
    for (int i = 0; i < num_candidates; ++i) {
        auto const& row_candidates = state.candidates.at(i);
        if (row_candidates.empty()) {
            continue;
        }
        auto const& best_cand = row_candidates.at(0);

        if (best_cand.cost < 0 &&
            !(state.node_markers.at(best_cand.node_id[0]) || state.node_markers.at(best_cand.node_id[1]))) {
            float node_width = db.node_size_x[best_cand.node_id[0]];
            float target_node_width = db.node_size_x[best_cand.node_id[1]];
            Space space = get_space(db, state, best_cand.node_id[0]);
            Space target_space = get_space(db, state, best_cand.node_id[1]);

            // space may no longer be large enough or the previously computed
            // locations may not be correct any more
            if (best_cand.node_xl[0][1] >= target_space.xl && best_cand.node_xl[0][1] + node_width <= target_space.xh &&
                best_cand.node_xl[1][1] >= space.xl && best_cand.node_xl[1][1] + target_node_width <= space.xh) {
                // state.node_markers[best_cand.node_id[0]] = 1;
                // state.node_markers[best_cand.node_id[1]] = 1;
                mark_dependent_nodes(db, state, best_cand.node_id[0], 1);
                mark_dependent_nodes(db, state, best_cand.node_id[1], 1);

                BinMapIndex& bin_id = state.node2bin_map.at(best_cand.node_id[0]);
                BinMapIndex& target_bin_id = state.node2bin_map.at(best_cand.node_id[1]);
                RowMapIndex& row_id = state.node2row_map.at(best_cand.node_id[0]);
                RowMapIndex& target_row_id = state.node2row_map.at(best_cand.node_id[1]);
                auto& row2nodes = state.row2node_map.at(row_id.row_id);
                auto& target_row2nodes = state.row2node_map.at(target_row_id.row_id);

                db.x[best_cand.node_id[0]] = best_cand.node_xl[0][1];
                db.y[best_cand.node_id[0]] = best_cand.node_yl[0][1];
                db.x[best_cand.node_id[1]] = best_cand.node_xl[1][1];
                db.y[best_cand.node_id[1]] = best_cand.node_yl[1][1];
                int& bin2node_map_node_id = state.bin2node_map.at(bin_id.bin_id).at(bin_id.sub_id);
                int& bin2node_map_target_node_id = state.bin2node_map.at(target_bin_id.bin_id).at(target_bin_id.sub_id);
                std::swap(bin2node_map_node_id, bin2node_map_target_node_id);
                std::swap(bin_id, target_bin_id);

                // update row2node_map and node2row_map
                std::swap(row2nodes[row_id.sub_id], target_row2nodes[target_row_id.sub_id]);
                std::swap(row_id, target_row_id);
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

    compute_search_bins(db, state, db.i_bgn, db.i_end);

    double time_run = 0;

    for (int i = db.i_bgn; i < db.i_end; i += state.batch_size) {
        // all results are stored in state.candidates
        int idx_bgn = i;
        int idx_end = std::min(i + state.batch_size, db.i_end);

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
        apply_candidates(db, state, idx_end - idx_bgn);
        apply_candidates_runs += 1;
    }

     logger.info("compute_candidate_cost takes %d runs", compute_candidate_cost_runs);
     logger.info("reduce_min_2d takes %d runs",reduce_min_2d_runs);
    logger.info("apply_candidates takes %d runs", apply_candidates_runs);
}  // END MODULE

//---------------------------------------------------------------------

void globalSwap(DetailedPlaceData& db, int num_bins_x, int num_bins_y, int max_iters, int batch_size) {
    logger.info("============= Running DP: Global Swap =============");
    db.num_bins_x = num_bins_x;
    db.num_bins_y = num_bins_y;
    db.bin_size_x = (db.xh - db.xl) / num_bins_x;
    db.bin_size_y = (db.yh - db.yl) / num_bins_y;

    SwapState state;
    state.num_threads = db.num_threads;

    const float stop_threshold = 0.1 / 100;
    state.batch_size = batch_size;  // TODO: #visible nodes
    int max_num_candidates_per_row = (2 << (int)log2(ceil(sqrt(db.num_nodes / (db.num_bins_x * db.num_bins_y)))));
    state.max_num_candidates =
        (1 << (int)ceil(log2(ceil(db.bin_size_y / db.row_height)))) * max_num_candidates_per_row * 5;
    state.max_num_candidates_all = state.batch_size * state.max_num_candidates;
    state.search_bin_strategy = 1;

    logger.info("#binX: %d, #binY: %d", num_bins_x, num_bins_y);
    logger.info("#Candidates: /row: %d, max: %d, all: %d",
                max_num_candidates_per_row,
                state.max_num_candidates,
                state.max_num_candidates_all);

    // distribute cells to rows
    state.row2node_map.resize(db.num_sites_y);
    state.node2row_map.resize(db.num_nodes);
    db.make_row2node_map(db.x, db.y, state.row2node_map);
    for (int i = 0; i < db.num_sites_y; ++i) {
        for (unsigned int j = 0; j < state.row2node_map[i].size(); ++j) {
            auto& row_id = state.node2row_map.at(state.row2node_map[i][j]);
            row_id.row_id = i;
            row_id.sub_id = j;
        }
    }

    // distribute cells to bin
    state.bin2node_map.resize(db.num_bins_x * db.num_bins_y);
    state.node2bin_map.resize(db.i_end);
    db.make_bin2node_map(db.x, db.y, db.node_size_x, db.node_size_y, state.bin2node_map, state.node2bin_map);

    state.ordered_nodes.resize(db.i_end);
    std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);

    state.candidates.resize(state.batch_size);
    state.search_bins.resize(db.i_end);
    state.net_hpwls.resize(db.num_nets);
    state.node_markers.assign(db.num_nodes, 0);

    std::vector<float> hpwls(max_iters + 1);
    // hpwls[0] = compute_total_hpwl(db, state, db.x, db.y, state.net_hpwls.data());
    hpwls[0] = db.compute_total_hpwl_concurrent(db.x, db.y, state.net_hpwls.data());

    logger.info("initial hpwl = %.3f", hpwls[0]);
    for (int iter = 0; iter < max_iters; ++iter) {
        std::random_shuffle(state.ordered_nodes.begin(), state.ordered_nodes.end());
        global_swap(db, state);

        // hpwls[iter + 1] = compute_total_hpwl(db, state, db.x, db.y, state.net_hpwls.data());
        hpwls[iter + 1] = db.compute_total_hpwl_concurrent(db.x, db.y, state.net_hpwls.data());
        logger.info("iteration %d: hpwl %.3f => %.3f (imp. %g%%)",
                    iter,
                    hpwls[0],
                    hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);
        state.search_bin_strategy = !state.search_bin_strategy;

        if ((iter & 1) && hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
            break;
        }
    }
}  // END MODULE

//---------------------------------------------------------------------
}  // namespace dp