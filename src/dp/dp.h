#pragma once

#include "global.h"
#include "parser/db/Database.h"
#include "parser/gp/GPDatabase.h"
#include "detailed_place_db.h"

#include "placer/database.h"
#include "placer/evaluator.h"

#include "abacus_legalize.h"
#include "greedy_legalize.h"
#include "global_swap.h"
#include "k_reorder.h"
#include "independent_set_matching.h"

#include "legality_check.h"
#include "macro_legalize.h"
#include "macro_floorplan.h"
#include "abacus_legalize_v2.h"
#include "greedy_legalize_v2.h"

#include "detailed_place_db_cross_chip.h"
#include "global_swap_cross_chip.h"
#include "via_legalize.h"
#include "via_detailed_place.h"
#include "fp/fp/floorplan.h"

namespace dp {

void legalizationV1(
    NodeData& data, torch::Tensor node_pos_lg, torch::Tensor node_size_lg, int cell_mov_lhs, int cell_mov_rhs);

void legalizationV2(NodeData& data,
                    DetailedPlaceDataTensor& lg_db_at,
                    torch::Tensor node_pos_lg,
                    int num_sites_y,
                    float row_height,
                    float row_start,
                    int num_bins_x,
                    int num_bins_y,
                    int cell_mov_lhs, 
                    int cell_mov_rhs,
                    bool only_macro = false);

void detail_placement(NodeData& data,
                      DetailedPlaceDataTensor& dp_db_at,
                      torch::Tensor node_pos_dp,
                      int num_sites_y,
                      float row_height,
                      int num_bins_x,
                      int num_bins_y,
                      bool via_dp = false);

}  // namespace dp
