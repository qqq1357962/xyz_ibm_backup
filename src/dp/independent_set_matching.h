#pragma once

#include "auction_cpu.h"
#include "detailed_place_db.h"
#include "diamond_search.h"
#include "global.h"

namespace dp {

struct IndependentSetMatchingState {
    std::vector<int> ordered_nodes;
    std::vector<std::vector<int>> independent_sets;
    std::vector<unsigned char> dependent_markers;
    std::vector<unsigned char> selected_markers;
    std::vector<int> num_selected_markers;
    std::vector<GridIndex<int>> search_grids;

    std::vector<std::vector<int>> bin2node_map;  ///< the first dimension is
                                                 ///< size, all the cells are
                                                 ///< categorized by width
    std::vector<BinMapIndex> node2bin_map;
    std::vector<Space> spaces;  ///< not used yet

    std::vector<std::vector<int>> cost_matrices;  ///< the convergence rate is related to numerical scale
    std::vector<std::vector<int>> solutions;
    std::vector<int> orig_costs;                   ///< original cost before matching
    std::vector<int> target_costs;                 ///< target cost after matching
    std::vector<std::vector<float>> target_pos_x;  ///< temporary storage of cell locations
    std::vector<std::vector<float>> target_pos_y;
    std::vector<std::vector<BinMapIndex>> target_node2bin_map;
    std::vector<std::vector<Space>> target_spaces;  ///< not used yet

    int batch_size;
    int set_size;
    int grid_size;
    int max_diamond_search_sequence;
    int num_moved;
    float large_number;
    float skip_threshold;  ///< ignore connections if cells are far apart
    int num_threads;
};

void independentSetMatching(DetailedPlaceData& db,
                            int num_bins_x,
                            int num_bins_y,
                            int batch_size = 2048,
                            int set_size = 128,
                            int max_iters = 50);

void independentSetMatchingSequential(
    DetailedPlaceData& db, int num_bins_x, int num_bins_y, int set_size = 128, int max_iters = 50);

}  // namespace dp