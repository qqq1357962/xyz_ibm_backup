#pragma once

#include <omp.h>

#include "../atomic_op.h"
#include "global.h"

torch::Tensor density_map_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_nodes,
                                  bool deterministic = true);

torch::Tensor macro_density_map_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_nodes,
                                  bool deterministic = true);

torch::Tensor density_map_backward(torch::Tensor normalize_node_info,
                                   torch::Tensor grad_mat,
                                   torch::Tensor sorted_node_map,
                                   torch::Tensor node_grad,
                                   torch::Tensor& node_grad_4part,
                                   torch::Tensor& macro_mask,
                                   float grad_weight,
                                   int num_bin_x,
                                   int num_bin_y,
                                   int num_nodes,
                                   bool deterministic = true);

at::Tensor density_map_forward_naive(at::Tensor node_pos,
                                     at::Tensor node_size,
                                     at::Tensor node_weight,
                                     at::Tensor unit_len,
                                     at::Tensor aux_mat,
                                     int num_bin_x,
                                     int num_bin_y,
                                     int num_nodes,
                                     float min_node_w,
                                     float min_node_h,
                                     float margin,
                                     bool clamp_node,
                                     bool deterministic = true);

at::Tensor density_map_normalize_node(torch::Tensor node_pos,
                                      torch::Tensor node_size,
                                      torch::Tensor node_weight,
                                      torch::Tensor expand_ratio,
                                      torch::Tensor unit_len,
                                      torch::Tensor normalize_node_info,
                                      int num_bin_x,
                                      int num_bin_y,
                                      int num_nodes);

torch::Tensor density_backward(torch::Tensor normalize_node_info,
                               torch::Tensor sorted_node_map,
                               torch::Tensor density_map,
                               int num_bin_x,
                               int num_bin_y,
                               int num_nodes);

torch::Tensor apply_kernel(torch::Tensor density_map,
                           torch::Tensor aux_mat,
                           int num_bin_x,
                           int num_bin_y,
                           torch::Tensor kernel,
                           int kernelWidth);