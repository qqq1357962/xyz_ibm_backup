#pragma once

#include "global.h"
#include "placer/database.h"
#include <map>
#include <queue>
#include <vector>

void viaDetailedPlace(const torch::Tensor node_pos,
                    const torch::Tensor node_size,
                    torch::Tensor node_pos_legal,
                    torch::Tensor node_weight,
                    torch::Tensor die_info,
                    torch::Tensor numRows,
                    torch::Tensor rowHeights,
                    int num_nodes,
                    torch::Tensor node_pos_all,
                    torch::Tensor node_die,
                    NodeData& data);