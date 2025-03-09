#pragma once

#include "detailed_place_db.h"
#include "global.h"
#include "utils.h"

namespace dp {

#define MAX_NUM_THREADS 128

struct KReorderInstance {
    int group_id;
    int row_id;
    int idx_bgn;
    int idx_end;
    // int permute_id;
};

struct KReorderState {
    std::vector<std::vector<int>> row2node_map;
    std::vector<std::vector<int>> permutations;
    std::vector<unsigned char> net_markers;
    std::vector<float> node_space_x;  ///< cell size with spaces
    std::vector<float> target_x[MAX_NUM_THREADS];
    std::vector<float> target_sizes[MAX_NUM_THREADS];
    std::vector<int> target_nodes[MAX_NUM_THREADS];

    std::vector<unsigned char> adjacency_matrix;  ///< adjacency matrix for row graph
    std::vector<std::vector<int>> row_graph;      ///< adjacency list for row graph
    std::vector<std::vector<int>> independent_rows;
    std::vector<std::vector<KReorderInstance>> reorder_instances;

    int K;
    int num_moved;
    int num_threads;
};

void compute_row_conflict_graph(const DetailedPlaceData& db,
                                const std::vector<std::vector<int>>& state_row2node_map,
                                std::vector<unsigned char>& state_adjacency_matrix,
                                std::vector<std::vector<int>>& state_row_graph,
                                int num_threads);

void compute_independent_rows(const DetailedPlaceData& db,
                              const std::vector<std::vector<int>>& state_row_graph,
                              std::vector<std::vector<int>>& state_independent_rows);

void compute_position(
    const DetailedPlaceData& db, KReorderState& state, int row_id, int idx_bgn, int idx_end, int permute_id);

float compute_reorder_hpwl(
    const DetailedPlaceData& db, KReorderState& state, int row_id, int idx_bgn, int idx_end, int permute_id);

void apply_reorder(DetailedPlaceData& db,
                   KReorderState& state,
                   int row_id,
                   int idx_bgn,
                   int idx_end,
                   const std::vector<int>& permutation,
                   const std::vector<float>& target_x);

void kReorder(DetailedPlaceData& db, int num_bins_x, int num_bins_y, int K = 4, int max_iters = 2);

} // namespace dp