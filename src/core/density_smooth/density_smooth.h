#pragma once

#include <omp.h>

#include "../atomic_op.h"
#include "global.h"

// torch::Tensor density_smooth_forward(torch::Tensor node_pos,
//                                        torch::Tensor node_size,
//                                        torch::Tensor node_weight,
//                                        torch::Tensor aux_mat,
//                                        torch::Tensor unit_len,
//                                        int num_bin_x,
//                                        int num_bin_y,
//                                        int num_nodes);

// torch::Tensor density_smooth_backward(torch::Tensor node_pos,
//                                         torch::Tensor node_size,
//                                         torch::Tensor grad_mat,
//                                         torch::Tensor unit_len,
//                                         torch::Tensor node_grad,
//                                         int num_bin_x,
//                                         int num_bin_y,
//                                         int num_nodes);

torch::Tensor density_smooth_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_nodes);

torch::Tensor density_smooth_backward(torch::Tensor normalize_node_info,
                                   torch::Tensor grad_mat,
                                   torch::Tensor sorted_node_map,
                                   torch::Tensor node_grad,
                                   float grad_weight,
                                   int num_bin_x,
                                   int num_bin_y,
                                   int num_nodes);

at::Tensor density_smooth_normalize_node(torch::Tensor node_pos,
                                      torch::Tensor node_size,
                                      torch::Tensor node_weight,
                                      torch::Tensor expand_ratio,
                                      torch::Tensor unit_len,
                                      torch::Tensor normalize_node_info,
                                      int num_bin_x,
                                      int num_bin_y,
                                      int num_nodes);