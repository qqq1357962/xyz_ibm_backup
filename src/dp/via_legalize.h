#pragma once

#include "global.h"
#include <string>
#include <map>
#include <stack>

void viaLegalization(const torch::Tensor node_pos,
                     const torch::Tensor node_size,
                     torch::Tensor node_pos_legal,
                     torch::Tensor node_weight,
                     torch::Tensor die_info,
                     torch::Tensor numRows,
                     torch::Tensor rowHeights,
                     int num_nodes);