#include <torch/torch.h>

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic);

torch::Tensor macro_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic);

// std::tuple<at::Tensor, at::Tensor> density_map_cuda_backward(torch::Tensor normalize_node_info,
torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        torch::Tensor& node_grad_4part,
                                        torch::Tensor& macro_mask,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        bool deterministic);

torch::Tensor density_map_cuda_forward_naive(torch::Tensor node_pos,
                                             torch::Tensor node_size,
                                             torch::Tensor node_weight,
                                             torch::Tensor unit_len,
                                             torch::Tensor aux_mat,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_nodes,
                                             float min_node_w,
                                             float min_node_h,
                                             float margin,
                                             bool clamp_node,
                                             bool deterministic);

torch::Tensor density_map_cuda_normalize_node(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor unit_len,
                                              torch::Tensor normalize_node_info,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_nodes);

torch::Tensor density_backward_cuda(torch::Tensor normalize_node_info,
                                    torch::Tensor sorted_node_map,
                                    torch::Tensor density_map,
                                    int num_bin_x,
                                    int num_bin_y,
                                    int num_nodes);

torch::Tensor apply_kernel_cuda(torch::Tensor density_map,
                                torch::Tensor aux_mat,
                                int num_bin_x,
                                int num_bin_y,
                                torch::Tensor kernel,
                                int kernelWidth);