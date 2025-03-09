#pragma once

#include "detailed_place_db.h"
#include "global.h"
#include "fp/fp/floorplan.h"
#include "fp/db/Database.h"

namespace dp {
// bool macroFloorplan(NodeData& data,
//                     DetailedPlaceDataTensor& lg_db_at,
//                     torch::Tensor node_pos_lg,
//                     int num_sites_y,
//                     float row_height,
//                     float row_start,
//                     int cell_mov_lhs, 
//                     int cell_mov_rhs);

bool macroFloorplan(
                    NodeData& data,
                    //DetailedPlaceDataTensor& lg_db_at,
                    torch::Tensor node_pos_lg,
                    torch::Tensor mov_node_weights,
                    torch::Tensor num_sites_y,
                    torch::Tensor row_height,
                    float row_start,
                    int cell_mov_lhs, 
                    int cell_mov_rhs);

}  // namespace dp
