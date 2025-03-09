#include "k_reorder.h"

#include <omp.h>

namespace dp {

void compute_row_conflict_graph(const DetailedPlaceData& db,
                                const std::vector<std::vector<int>>& state_row2node_map,
                                std::vector<unsigned char>& state_adjacency_matrix,
                                std::vector<std::vector<int>>& state_row_graph,
                                int num_threads) {
    // adjacency matrix
    state_adjacency_matrix.assign(db.num_sites_y * db.num_sites_y, 0);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int net_id = 0; net_id < db.num_nets; ++net_id) {
        if (db.net_mask[net_id]) {
            int net2pin_start = db.flat_net2pin_start_map[net_id];
            int net2pin_end = db.flat_net2pin_start_map[net_id + 1];
            for (int net2pin_id1 = net2pin_start; net2pin_id1 < net2pin_end; ++net2pin_id1) {
                int net_pin_id1 = db.flat_net2pin_map[net2pin_id1];
                int node_id1 = db.pin2node_map[net_pin_id1];
                if (db.node_weight[node_id1] == 0) continue;
                if (node_id1 < db.i_end) {
                    // FIXME: float divide
                    // int row_id1 = floorDiv(db.y[node_id1] - db.yl, db.row_height);
                    int row_id1 = floorDivRound(db.y[node_id1] - db.yl, db.row_height, db.site_width_safe_divide);
                    row_id1 = std::min(std::max(row_id1, 0), db.num_sites_y - 1);
                    for (int net2pin_id2 = net2pin_id1; net2pin_id2 < net2pin_end; ++net2pin_id2) {
                        int net_pin_id2 = db.flat_net2pin_map[net2pin_id2];
                        int node_id2 = db.pin2node_map[net_pin_id2];
                        if (db.node_weight[node_id2] == 0) continue;
                        if (node_id2 < db.i_end) {
                            // int row_id2 = floorDiv(db.y[node_id2] - db.yl, db.row_height);
                            int row_id2 =
                                floorDivRound(db.y[node_id2] - db.yl, db.row_height, db.site_width_safe_divide);
                            row_id2 = std::min(std::max(row_id2, 0), db.num_sites_y - 1);
                            unsigned char& adjacency_matrix_element1 =
                                state_adjacency_matrix.at(row_id1 * db.num_sites_y + row_id2);
                            unsigned char& adjacency_matrix_element2 =
                                state_adjacency_matrix.at(row_id2 * db.num_sites_y + row_id1);
                            if (!adjacency_matrix_element1) {
#pragma omp atomic
                                adjacency_matrix_element1 |= 1;
                            }
                            if (!adjacency_matrix_element2) {
#pragma omp atomic
                                adjacency_matrix_element2 |= 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // adjacency list
    state_row_graph.assign(db.num_sites_y, std::vector<int>());
#pragma omp parallel for num_threads(num_threads)
    for (int row_id = 0; row_id < db.num_sites_y; ++row_id) {
        auto& adjacency_vec = state_row_graph[row_id];
        for (int other_row_id = 0; other_row_id < db.num_sites_y; ++other_row_id) {
            if (row_id != other_row_id && state_adjacency_matrix.at(row_id * db.num_sites_y + other_row_id)) {
                adjacency_vec.push_back(other_row_id);
            }
        }
    }
}

void compute_independent_rows(const DetailedPlaceData& db,
                              const std::vector<std::vector<int>>& state_row_graph,
                              std::vector<std::vector<int>>& state_independent_rows) {
    // generate independent sets of rows
    std::vector<unsigned char> dependent_markers(db.num_sites_y, 0);
    std::vector<unsigned char> selected_markers(db.num_sites_y, 0);
    int num_selected = 0;
    while (num_selected < db.num_sites_y) {
        std::vector<int> independent_rows;
        for (int row_id = 0; row_id < db.num_sites_y; ++row_id) {
            if (!dependent_markers[row_id] && !selected_markers[row_id]) {
                independent_rows.push_back(row_id);
                dependent_markers[row_id] = 1;
                selected_markers[row_id] = 1;
                num_selected += 1;

                for (auto other_row_id : state_row_graph[row_id]) {
                    dependent_markers[other_row_id] = 1;
                }
            }
        }
        // recover marker
        for (auto i : independent_rows) {
            for (auto other_row_id : state_row_graph[i]) {
                dependent_markers[other_row_id] = 0;
            }
        }
        state_independent_rows.push_back(independent_rows);
    }
}

void compute_position(
    const DetailedPlaceData& db, KReorderState& state, int row_id, int idx_bgn, int idx_end, int permute_id) {
    auto row2nodes = state.row2node_map.at(row_id).data() + idx_bgn;
    auto const& permutation = state.permutations.at(permute_id);
    int K = idx_end - idx_bgn;

    int tid = omp_get_thread_num();
    auto& target_x = state.target_x[tid];
    auto& target_sizes = state.target_sizes[tid];

    target_x.resize(idx_end - idx_bgn);
    target_sizes.resize(idx_end - idx_bgn);

    // find left boundary
    if (K) {
        int node_id = row2nodes[0];
        target_x.at(0) = db.x[node_id];
    }
    // record sizes, and pack to left
    for (int i = 0; i < K; ++i) {
        int node_id = row2nodes[i];
        assert(node_id < db.i_end);
        target_sizes[permutation.at(i)] = state.node_space_x[node_id];
    }
    for (int i = 1; i < K; ++i) {
        target_x[i] = target_x[i - 1] + target_sizes[i - 1];
    }
}

float compute_reorder_hpwl(
    const DetailedPlaceData& db, KReorderState& state, int row_id, int idx_bgn, int idx_end, int permute_id) {
    auto const& row2nodes = state.row2node_map.at(row_id);
    auto const& permutation = state.permutations.at(permute_id);

    for (int i = 0; i < idx_end - idx_bgn; ++i) {
        if (permutation.at(i) >= idx_end - idx_bgn) {
            return std::numeric_limits<float>::max();
        }
    }

    compute_position(db, state, row_id, idx_bgn, idx_end, permute_id);

    int tid = omp_get_thread_num();
    auto& target_x = state.target_x[tid];

    float cost = 0;
    if (!db.via_dp) {
        for (int i = idx_bgn; i < idx_end; ++i) {
            int node_id = row2nodes.at(i);
            if (db.node_weight[node_id] == 0) continue;
            for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                 node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
                int node_pin_id = db.flat_node2pin_map[node2pin_id];
                int net_id = db.pin2net_map[node_pin_id];
                if (db.net_mask[net_id] && !state.net_markers[net_id]) {
                    float bxl = db.xh;
                    float bxh = db.xl;
                    for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                         net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
                        int net_pin_id = db.flat_net2pin_map[net2pin_id];
                        int other_node_id = db.pin2node_map[net_pin_id];
                        if (db.node_die[other_node_id] == 0) continue;
                        float other_node_xl;
                        auto found = std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, other_node_id);
                        if (found != row2nodes.begin() + idx_end) {
                            int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                            int permuted_offset = permutation.at(distance);
                            // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                            other_node_xl = target_x.at(permuted_offset);
                        } else {
                            other_node_xl = db.x[other_node_id];
                        }
                        other_node_xl += db.pin_offset_x[net_pin_id];
                        bxl = std::min(bxl, other_node_xl);
                        bxh = std::max(bxh, other_node_xl);
                    }
                    cost += bxh - bxl;
                    state.net_markers[net_id] = 1;
                }
            }
        }
    } else {
        for (int i = idx_bgn; i < idx_end; ++i) {
            int node_id = row2nodes.at(i);
            if (db.node_weight[node_id] == 0) continue;
            for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                 node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
                int node_pin_id = db.flat_node2pin_map[node2pin_id];
                int net_id = db.pin2net_map[node_pin_id];
                if (db.net_mask[net_id] && !state.net_markers[net_id]) {
                    float bxls[2] = {db.xh, db.xh};
                    float bxhs[2] = {db.xl, db.xl};
                    for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                         net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
                        int net_pin_id = db.flat_net2pin_map[net2pin_id];
                        int other_node_id = db.pin2node_map[net_pin_id];
                        if ((other_node_id < db.num_movable_nodes) ||
                            ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
                            float other_node_xl;
                            auto found =
                                std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, other_node_id);
                            if (found != row2nodes.begin() + idx_end) {
                                int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                                int permuted_offset = permutation.at(distance);
                                // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                                other_node_xl = target_x.at(permuted_offset);
                            } else {
                                other_node_xl = db.x[other_node_id];
                            }

                            other_node_xl += db.pin_offset_x[net_pin_id];
                            int c_id = db.node_die[other_node_id];
                            if (c_id != 2) {
                                bxls[c_id] = std::min(bxls[c_id], other_node_xl);
                                bxhs[c_id] = std::max(bxhs[c_id], other_node_xl);
                            } else {
                                for (c_id = 0; c_id < 2; c_id++) {
                                    bxls[c_id] = std::min(bxls[c_id], other_node_xl);
                                    bxhs[c_id] = std::max(bxhs[c_id], other_node_xl);
                                }
                            }
                        }
                    }
                    cost += (bxhs[0] - bxls[0]) + (bxhs[1] - bxls[1]);
                    state.net_markers[net_id] = 1;
                }
            }
        }
    }

    for (int i = idx_bgn; i < idx_end; ++i) {
        int node_id = row2nodes.at(i);
        if (db.node_weight[node_id] == 0) continue;
        for (int node2pin_id = db.flat_node2pin_start_map[node_id];
             node2pin_id < db.flat_node2pin_start_map[node_id + 1];
             ++node2pin_id) {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            state.net_markers[net_id] = 0;
        }
    }
    return cost;
}

void apply_reorder(DetailedPlaceData& db,
                   KReorderState& state,
                   int row_id,
                   int idx_bgn,
                   int idx_end,
                   const std::vector<int>& permutation,
                   const std::vector<float>& target_x) {
    auto row2nodes = state.row2node_map.at(row_id).data() + idx_bgn;
    int K = idx_end - idx_bgn;

    int tid = omp_get_thread_num();
    auto& target_nodes = state.target_nodes[tid];
    target_nodes.resize(K);

    for (int i = 0; i < K; ++i) {
        int node_id = row2nodes[i];
        target_nodes.at(i) = node_id;
    }

    for (int i = 0; i < K; ++i) {
        int node_id = row2nodes[i];
        float xx = target_x.at(permutation.at(i));
        if (db.x[node_id] != xx) {
            state.num_moved += 1;
        }
        db.x[node_id] = xx;
    }

    for (int i = 0; i < K; ++i) {
        row2nodes[permutation.at(i)] = target_nodes.at(i);
    }
}
int compute_best_reorder(
    const DetailedPlaceData& db, KReorderState& state, int row_id, int idx_bgn, int idx_end, std::vector<float> &best_target_x_tid, int max_permute) {
    float best_cost = std::numeric_limits<float>::max();
    std::unordered_map<int, vector<vector<float>>> netID2bound_map;
    std::unordered_map<int, vector<int>> netID2pinID_map;
    int best_permute_id = std::numeric_limits<int>::max();
    
    std::vector<int> permute_node;
    
    for(int permute_id = 0; permute_id < max_permute; permute_id++) {
        auto const& row2nodes = state.row2node_map.at(row_id);
        auto const& permutation = state.permutations.at(permute_id);
        int invalid = 0;
    
        for (int i = 0; i < idx_end - idx_bgn; ++i) {
            if (permutation.at(i) >= idx_end - idx_bgn) {
                invalid = 1;
            }
        }
        
        if(invalid == 1) {
            continue;
        }
    
        compute_position(db, state, row_id, idx_bgn, idx_end, permute_id);
    
        int tid = omp_get_thread_num();
        auto& target_x = state.target_x[tid];
    
        float cost = 0;
        if (!db.via_dp) {
            if(permute_id == 0) {
                for (int i = idx_bgn; i < idx_end; ++i) {
                    int node_id = row2nodes.at(i);
                    if (db.node_weight[node_id] == 0) continue;
                    for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                         node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
                        int node_pin_id = db.flat_node2pin_map[node2pin_id];
                        int net_id = db.pin2net_map[node_pin_id];
                        if (db.net_mask[net_id] && !state.net_markers[net_id]) {
                            float bxl = db.xh;
                            float bxh = db.xl;
                            float bxl_not_permute_node = db.xh;
                            float bxh_not_permute_node = db.xl;
                            vector<int> permute_node_pin;
                            for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                                net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
                                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                                int other_node_id = db.pin2node_map[net_pin_id];
                                if (db.node_die[other_node_id] == 0) continue;
                                float other_node_xl;
                                auto found = std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, other_node_id);
                                if (found != row2nodes.begin() + idx_end) {
                                    int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                                    int permuted_offset = permutation.at(distance);
                                    // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                                    other_node_xl = target_x.at(permuted_offset);
                                    permute_node_pin.push_back(net_pin_id);
                                } else {
                                    other_node_xl = db.x[other_node_id];
                                }
                                other_node_xl += db.pin_offset_x[net_pin_id];
                                bxl = std::min(bxl, other_node_xl);
                                bxh = std::max(bxh, other_node_xl);
                                if (!(found != row2nodes.begin() + idx_end)) {
                                    bxl_not_permute_node = std::min(bxl_not_permute_node, other_node_xl);
                                    bxh_not_permute_node = std::max(bxh_not_permute_node, other_node_xl);
                                }
                            }
                            cost += bxh - bxl;
                            state.net_markers[net_id] = 1;
                            vector<float> tmp;
                            tmp.push_back(bxl_not_permute_node);
                            tmp.push_back(bxh_not_permute_node);
                            vector<vector<float>> tmp2d;
                            tmp2d.push_back(tmp);
                            netID2bound_map.insert({net_id, tmp2d});
                            netID2pinID_map.insert({net_id, permute_node_pin});

                        }
                    }
                }
            } else {
                for(auto kv : netID2pinID_map) {
                    float bxl = db.xh;
                    float bxh = db.xl;
                    bxl = std::min(netID2bound_map.at(kv.first)[0][0], bxl);
                    bxh = std::max(netID2bound_map.at(kv.first)[0][1], bxh);
                    for(int i = 0; i < kv.second.size(); i++) {
                        int net_pin_id = kv.second[i];
                        int node_id = db.pin2node_map[net_pin_id];
                        if (db.node_die[node_id] == 0) continue;
                        float node_xl;
                        auto found = std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, node_id);
                        if (found != row2nodes.begin() + idx_end) {
                            int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                            int permuted_offset = permutation.at(distance);
                            // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                            node_xl = target_x.at(permuted_offset);
                        } else {
                            continue;
                        }
                        node_xl += db.pin_offset_x[net_pin_id];
                        bxl = std::min(bxl, node_xl);
                        bxh = std::max(bxh, node_xl);
                    }
                    cost += bxh - bxl; 
                }
            }
        } else {
            if(permute_id == 0){
                for (int i = idx_bgn; i < idx_end; ++i) {
                    int node_id = row2nodes.at(i);
                    if (db.node_weight[node_id] == 0) continue;
                    for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                         node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
                        int node_pin_id = db.flat_node2pin_map[node2pin_id];
                        int net_id = db.pin2net_map[node_pin_id];
                        if (db.net_mask[net_id] && !state.net_markers[net_id]) {
                            vector<int> permute_node_pin;
                            float bxls[2] = {db.xh, db.xh};
                            float bxhs[2] = {db.xl, db.xl};
                            float bxl_not_permute_node[2] = {db.xh, db.xh};
                            float bxh_not_permute_node[2] = {db.xl, db.xl};
                            for (int net2pin_id = db.flat_net2pin_start_map[net_id];
                                 net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
                                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                                int other_node_id = db.pin2node_map[net_pin_id];
                                if ((other_node_id < db.num_movable_nodes) ||
                                    ((other_node_id >= db.num_movable_nodes) && (db.node_weight[other_node_id] == 1))) {
                                    float other_node_xl;
                                    auto found =
                                        std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, other_node_id);
                                    if (found != row2nodes.begin() + idx_end) {
                                        int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                                        int permuted_offset = permutation.at(distance);
                                        // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                                        other_node_xl = target_x.at(permuted_offset);
                                        permute_node_pin.push_back(net_pin_id);
                                    } else {
                                        other_node_xl = db.x[other_node_id];
                                    }
        
                                    other_node_xl += db.pin_offset_x[net_pin_id];
                                    int c_id = db.node_die[other_node_id];
                                    if (c_id != 2) {
                                        bxls[c_id] = std::min(bxls[c_id], other_node_xl);
                                        bxhs[c_id] = std::max(bxhs[c_id], other_node_xl);
                                    } else {
                                        for (c_id = 0; c_id < 2; c_id++) {
                                            bxls[c_id] = std::min(bxls[c_id], other_node_xl);
                                            bxhs[c_id] = std::max(bxhs[c_id], other_node_xl);
                                        }
                                    }
                                    if (!(found != row2nodes.begin() + idx_end)) {
                                        if (c_id != 2) {
                                            bxl_not_permute_node[c_id] = std::min(bxl_not_permute_node[c_id], other_node_xl);
                                            bxh_not_permute_node[c_id] = std::max(bxh_not_permute_node[c_id], other_node_xl);
                                        } else {
                                            for (c_id = 0; c_id < 2; c_id++) {
                                                bxl_not_permute_node[c_id] = std::min(bxl_not_permute_node[c_id], other_node_xl);
                                                bxh_not_permute_node[c_id] = std::max(bxh_not_permute_node[c_id], other_node_xl);
                                            }
                                        }
                                    }

                                }
                            }
                            cost += (bxhs[0] - bxls[0]) + (bxhs[1] - bxls[1]);
                            state.net_markers[net_id] = 1;
                            vector<float> tmp;
                            tmp.push_back(bxl_not_permute_node[0]);
                            tmp.push_back(bxh_not_permute_node[0]);
                            vector<float> tmp2;
                            tmp2.push_back(bxl_not_permute_node[1]);
                            tmp2.push_back(bxh_not_permute_node[1]);
                            vector<vector<float>> tmp2d;
                            tmp2d.push_back(tmp);
                            tmp2d.push_back(tmp2);
                            netID2bound_map.insert({net_id, tmp2d});
                            netID2pinID_map.insert({net_id, permute_node_pin});
                        }
                    }
                }
            } else {
                for(auto kv : netID2pinID_map) {
                    float bxls[2] = {db.xh, db.xh};
                    float bxhs[2] = {db.xl, db.xl};
                    for (int c_id = 0; c_id < 2; c_id++) {
                        bxls[c_id] = std::min(netID2bound_map.at(kv.first)[c_id][0], bxls[c_id]);
                        bxhs[c_id] = std::max(netID2bound_map.at(kv.first)[c_id][1], bxhs[c_id]);
                    }
                    for(int i = 0; i < kv.second.size(); i++) {
                        int net_pin_id = kv.second[i];
                        int node_id = db.pin2node_map[net_pin_id];
                        if (db.node_die[node_id] == 0) continue;
                        if ((node_id < db.num_movable_nodes) ||
                            ((node_id >= db.num_movable_nodes) && (db.node_weight[node_id] == 1))) {
                            float node_xl;
                            auto found = std::find(row2nodes.begin() + idx_bgn, row2nodes.begin() + idx_end, node_id);
                            if (found != row2nodes.begin() + idx_end) {
                                int distance = std::distance(row2nodes.begin() + idx_bgn, found);
                                int permuted_offset = permutation.at(distance);
                                // int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                                node_xl = target_x.at(permuted_offset);
                            } else {
                                continue;
                            }
                            int c_id = db.node_die[node_id];
                            node_xl += db.pin_offset_x[net_pin_id];
                            if (c_id != 2) {
                                bxls[c_id] = std::min(bxls[c_id], node_xl);
                                bxhs[c_id] = std::max(bxhs[c_id], node_xl);
                            } else {
                                for (c_id = 0; c_id < 2; c_id++) {
                                    bxls[c_id] = std::min(bxls[c_id], node_xl);
                                    bxhs[c_id] = std::max(bxhs[c_id], node_xl);
                                }
                            }
                        }
                    }
                    cost += (bxhs[0] - bxls[0]) + (bxhs[1] - bxls[1]);    
                }
            }
        }
    
        for (int i = idx_bgn; i < idx_end; ++i) {
            int node_id = row2nodes.at(i);
            if (db.node_weight[node_id] == 0) continue;
            for (int node2pin_id = db.flat_node2pin_start_map[node_id];
                 node2pin_id < db.flat_node2pin_start_map[node_id + 1];
                 ++node2pin_id) {
                int node_pin_id = db.flat_node2pin_map[node2pin_id];
                int net_id = db.pin2net_map[node_pin_id];
                state.net_markers[net_id] = 0;
            }
        }
        if(cost < best_cost) {
            best_permute_id = permute_id;
            best_cost = cost;
            best_target_x_tid = target_x;
        }
    }
    return best_permute_id;
}
void kReorder(DetailedPlaceData& db, int num_bins_x, int num_bins_y, int K, int max_iters) {
    logger.info("============= Running DP: %d-reorder =============", K);
    db.num_bins_x = num_bins_x;
    db.num_bins_y = num_bins_y;
    db.bin_size_x = (db.xh - db.xl) / num_bins_x;
    db.bin_size_y = (db.yh - db.yl) / num_bins_y;

    float stop_threshold = 0.1 / 100;
    int num_threads = db.num_threads;

    KReorderState state;
    state.K = K;
    state.num_threads = std::min(std::max(num_threads, 1), MAX_NUM_THREADS);

    // divide layout into rows
    // distribute cells into them
    state.row2node_map.resize(db.num_sites_y);
    // map node index to its location in row2node_map
    // we can compute the rows, so only the index within a row of row2node_map is
    // stored
    // state.node2row_map.resize(db.num_nodes);

    // distribute cells to rows
    db.make_row2node_map(db.x, db.y, state.row2node_map);

    state.node_space_x.resize(db.i_end);
    for (int i = 0; i < db.num_sites_y; ++i) {
        for (unsigned int j = 0; j < state.row2node_map.at(i).size(); ++j) {
            int node_id = state.row2node_map[i][j];
            if (db.node_weight[node_id] == 0) continue;
            if (node_id < db.i_end) {
                auto& space = state.node_space_x[node_id];
                float space_xl = db.x[node_id];
                float space_xh = db.xh;
                if (j + 1 < state.row2node_map[i].size()) {
                    int right_node_id = state.row2node_map[i][j + 1];
                    if (db.node_weight[right_node_id] == 0) continue;
                    space_xh = std::min(space_xh, db.x[right_node_id]);
                }
                space = space_xh - space_xl;
                // align space to sites, as I assume space_xl aligns to sites
                // I also assume node width should be integral numbers of sites
                space = floorDiv(space, db.site_width) * db.site_width;

                // cout << space << " | " << db.node_size_x[node_id] << endl;
                // cout << space << " | " << db.site_width << endl;
                // cout << space << " | " << db.node_size_x[node_id] << endl;
                assert_msg(space >= db.node_size_x[node_id],
                           "space %g, node_size_x[%d] %g, original space (%g, "
                           "%g), site_width %g",
                           space,
                           node_id,
                           db.node_size_x[node_id],
                           space_xl,
                           space_xh,
                           db.site_width);
            }
        }
    }

    state.permutations = quick_perm(K);
    state.net_markers.assign(db.num_nets, 0);

    compute_row_conflict_graph(db, state.row2node_map, state.adjacency_matrix, state.row_graph, num_threads);
    compute_independent_rows(db, state.row_graph, state.independent_rows);


    std::vector<float> best_target_x[MAX_NUM_THREADS];

    // count number of movement
    state.num_moved = 0;
    float hpwls[max_iters + 1];
    hpwls[0] = db.compute_total_hpwl();
    logger.info("initial hpwl = %.3f", hpwls[0]);

    for (int iter = 0; iter < max_iters; ++iter) {
        for (unsigned int group_id = 0; group_id < state.independent_rows.size(); ++group_id) {
            auto const& independent_rows = state.independent_rows[group_id];
            unsigned int num_independent_rows = independent_rows.size();
#pragma omp parallel for num_threads(state.num_threads) schedule(dynamic, 1)
            for (unsigned int group_row_id = 0; group_row_id < num_independent_rows; ++group_row_id) {
                int tid = omp_get_thread_num();
                auto& target_x = state.target_x[tid];
                auto& best_target_x_tid = best_target_x[tid];

                int row_id = independent_rows.at(group_row_id);
                auto const& row2nodes = state.row2node_map.at(row_id);
                for (int sub_id = 0; sub_id < (int)row2nodes.size(); sub_id += K / 2) {
                    int idx_bgn = sub_id;
                    int idx_end = std::min(sub_id + K, (int)row2nodes.size());
                    // stop at fixed cells and multi-row height cells
                    for (int i = idx_bgn; i < idx_end; ++i) {
                        int node_id = row2nodes.at(i);
                        if (node_id >= db.i_end || db.node_size_y[node_id] > db.row_height) {
                            idx_end = i;
                            break;
                        }
                    }
                    if (idx_end - idx_bgn < 2) {
                        continue;
                    }
                    float best_cost = std::numeric_limits<float>::max();
                    int best_pi = std::numeric_limits<int>::max();
                    /*
                    for (unsigned int pi = 0; pi < state.permutations.size(); ++pi) {
                        float cost = compute_reorder_hpwl(db, state, row_id, idx_bgn, idx_end, pi);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_pi = pi;
                            best_target_x_tid = target_x;
                        }
                    }
                    */
                   best_pi = compute_best_reorder(db, state, row_id, idx_bgn, idx_end, best_target_x_tid, state.permutations.size());

                    apply_reorder(
                        db, state, row_id, idx_bgn, idx_end, state.permutations.at(best_pi), best_target_x_tid);
                }
            }
        }

        hpwls[iter + 1] = db.compute_total_hpwl();
        logger.info("iteration %d: hpwl %.3f => %.3f (imp. %g%%)",
                    iter,
                    hpwls[0],
                    hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);

        if ((iter & 1) && hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
            break;
        }
    }
}

}  // namespace dp