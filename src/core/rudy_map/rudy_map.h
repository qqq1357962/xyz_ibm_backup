
#pragma once

#include <omp.h>

#include "../atomic_op.h"
#include "global.h"
#include "placer/database.h"

void rudy_map_forward_naive(PlaceData& data,
                            torch::Tensor pin_pos,
                            torch::Tensor unit_len,
                            torch::Tensor& horizontal_map,
                            torch::Tensor& vertical_map,
                            int num_bin_x,
                            int num_bin_y,
                            float margin);

void pin_density_map_forward_naive(PlaceData& data,
                                   torch::Tensor pin_pos,
                                   torch::Tensor unit_len,
                                   torch::Tensor& aux_mat,
                                   int num_bin_x,
                                   int num_bin_y,
                                   float margin);