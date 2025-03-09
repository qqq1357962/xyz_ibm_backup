#include <torch/torch.h>

torch::Tensor density_smooth_cuda_forward(torch::Tensor normalize_node_info,
                                          torch::Tensor sorted_node_map,
                                          torch::Tensor aux_mat,
                                          int num_bin_x,
                                          int num_bin_y,
                                          int num_bin_z,
                                          int num_nodes);

torch::Tensor density_smooth_cuda_backward(torch::Tensor normalize_node_info,
                                           torch::Tensor grad_mat,
                                           torch::Tensor sorted_node_map,
                                           torch::Tensor node_grad,
                                           float grad_weight,
                                           int num_bin_x,
                                           int num_bin_y,
                                           int num_bin_z,
                                           int num_nodes);

torch::Tensor density_smooth_cuda_normalize_node(torch::Tensor node_pos,
                                                 torch::Tensor node_size,
                                                 torch::Tensor node_weight,
                                                 torch::Tensor expand_ratio,
                                                 torch::Tensor unit_len,
                                                 torch::Tensor normalize_node_info,
                                                 int num_bin_x,
                                                 int num_bin_y,
                                                 int num_bin_z,
                                                 int num_nodes);