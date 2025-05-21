#include <torch/torch.h>

namespace GP3D {

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_bin_z,
                                       int num_nodes);

torch::Tensor macro_overlay_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor aux_mat,
                                        torch::Tensor node_rotate_grad,
                                        torch::Tensor rotate_rate,
                                        torch::Tensor unit_len,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_bin_z,
                                        int num_nodes,
                                        int num_macros);                                      

torch::Tensor macro_density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_bin_z,
                                       int num_macros,
                                       int num_nodes);

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        torch::Tensor rotate_state,
                                        torch::Tensor unit_len,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_bin_z,
                                        int num_nodes,
                                        int num_macros);

torch::Tensor density_map_cuda_forward_naive(torch::Tensor node_pos,
                                             torch::Tensor node_size,
                                             torch::Tensor node_weight,
                                             torch::Tensor unit_len,
                                             torch::Tensor aux_mat,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_bin_z,
                                             int num_nodes,
                                             float min_node_w,
                                             float min_node_h,
                                             float margin,
                                             bool clamp_node);

torch::Tensor density_map_cuda_normalize_node(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor unit_len,
                                              torch::Tensor normalize_node_info,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_bin_z,
                                              int num_nodes);

torch::Tensor density_backward_cuda(torch::Tensor normalize_node_info,
                                    torch::Tensor sorted_node_map,
                                    torch::Tensor density_map,
                                    int num_bin_x,
                                    int num_bin_y,
                                    int num_bin_z,
                                    int num_nodes);

torch::Tensor apply_kernel_cuda(torch::Tensor density_map,
                                torch::Tensor aux_mat,
                                int num_bin_x,
                                int num_bin_y,
                                int num_bin_z,
                                torch::Tensor kernel,
                                int kernelWidth);

}  // namespace GP3D