#pragma once

#include "global.h"
#include "detailed_place_db_cross_chip.h"
#include "gr/ViaData.h"

namespace dpc {

struct SwapCandidate {
    float cost;
    float node_xl[2][2];  ///< [0][] for node, [1][] for target node, [][0] for old,
                          ///< [][1] for new
    float node_yl[2][2];
    int node_id[2];  ///< [0] for node, [1] for target node

    vector<int> switch_list1;
    vector<int> switch_list2;
};

struct SwapState {
    std::vector<int> ordered_nodes;

    std::vector<std::vector<int>> row2node_map;
    std::vector<RowMapIndex> node2row_map;

    std::vector<std::vector<int>> bin2node_map;
    std::vector<BinMapIndex> node2bin_map;

    std::vector<std::vector<std::vector<int>>> row2node_maps;
    std::vector<std::vector<std::vector<int>>> bin2node_maps;

    std::vector<int> search_bins;
    int search_bin_strategy;  ///< how to compute search bins for eahc cell: 0 for
                              ///< cell bin, 1 for optimal region

    std::vector<std::vector<SwapCandidate>> candidates;

    std::vector<float> net_hpwls;             ///< HPWL for each net
    std::vector<unsigned char> node_markers;  ///< markers for cells

    int batch_size;
    int max_num_candidates;
    int max_num_candidates_all;
    int num_threads;
};

void globalSwapCrossChip(DetailedPlaceData& db,
                         const torch::Tensor node_swap,
                         torch::Tensor numRows,
                         torch::Tensor rowHeights,
                         int num_bins_x,
                         int num_bins_y,
                         int max_iters = 1,
                         int batch_size = 1);

void compute_search_bins(const DetailedPlaceData& db, SwapState& state, int begin, int end);
Space get_space(const DetailedPlaceData& db, const SwapState& state, int node_id);
void reset_state(DetailedPlaceData& db, SwapState& state);
float compute_positions_hint(const DetailedPlaceData& db,
                             const SwapState& state,
                             SwapCandidate& cand,
                             float node_xl,
                             float node_yl,
                             float node_width,
                             const Space& space);
void collect_candidates(const DetailedPlaceData& db, SwapState& state, int idx_bgn, int idx_end);

float compute_pair_hpwl_general(const DetailedPlaceData& db,
                                const SwapState& state,
                                int node_id,
                                float node_xl,
                                float node_yl,
                                int target_node_id,
                                float target_node_xl,
                                float target_node_yl,
                                // int c_id_prime,
                                int skip_node_id);
void compute_candidate_cost(DetailedPlaceData& db, SwapState& state);
void apply_candidates(DetailedPlaceData& db, SwapState& state, int num_candidates, int& count);

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void mark_dependent_nodes(const DetailedPlaceDBType& db,
                          IndependentSetMatchingStateType& state,
                          int node_id,
                          unsigned char value);
bool get_swap_net_info(DetailedPlaceData& db, int node_id, int target_node_id);

void recover_swap(DetailedPlaceData& db, vector<int>& switch_list);

bool check_via_space(DetailedPlaceData& db, int node_id, float node_xl, float node_yl, int net_id);  // TODO:

}  // namespace dpc