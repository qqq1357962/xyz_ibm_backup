#pragma once

#include "core/hpwl/hpwl.h"
#include "run_placement.h"

void post_process(PlaceData& data,
                  torch::Tensor node_pos,
                  torch::Tensor node_die,
                  torch::Tensor via_pos,
                  torch::Tensor via_size,
                  torch::Tensor row_shift,
                  torch::Tensor numRows,
                  torch::Tensor row_height);

void viaPlaceCentre(PlaceData& data,
                    torch::Tensor node_pos,
                    torch::Tensor node_die,
                    torch::Tensor via_pos,
                    torch::Tensor via_size,
                    torch::Tensor row_shift,
                    torch::Tensor numRows,
                    torch::Tensor row_height);