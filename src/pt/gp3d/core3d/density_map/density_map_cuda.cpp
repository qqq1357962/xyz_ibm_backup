#include "density_map_cuda.cuh"

#include "density_map.h"

namespace GP3D {

torch::Tensor density_map_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_bin_z,
                                  int num_nodes) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(aux_mat);

    return density_map_cuda_forward(
        normalize_node_info, sorted_node_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, num_nodes);
}

torch::Tensor macro_density_map_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_bin_z,
                                  int num_macros,
                                  int num_nodes) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(aux_mat);

    return macro_density_map_cuda_forward(
        normalize_node_info, sorted_node_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, num_macros, num_nodes);
}

torch::Tensor density_map_backward(torch::Tensor normalize_node_info,
                                   torch::Tensor grad_mat,
                                   torch::Tensor sorted_node_map,
                                   torch::Tensor node_grad,
                                   float grad_weight,
                                   int num_bin_x,
                                   int num_bin_y,
                                   int num_bin_z,
                                   int num_nodes) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(grad_mat);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(node_grad);

    return density_map_cuda_backward(normalize_node_info,
                                     grad_mat,
                                     sorted_node_map,
                                     node_grad,
                                     grad_weight,
                                     num_bin_x,
                                     num_bin_y,
                                     num_bin_z,
                                     num_nodes);
}

at::Tensor density_map_forward_naive(at::Tensor node_pos,
                                     at::Tensor node_size,
                                     at::Tensor node_weight,
                                     at::Tensor unit_len,
                                     at::Tensor aux_mat,
                                     int num_bin_x,
                                     int num_bin_y,
                                     int num_bin_z,
                                     int num_nodes,
                                     float min_node_w,
                                     float min_node_h,
                                     float margin,
                                     bool clamp_node) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_size);
    CHECK_INPUT(node_weight);
    CHECK_INPUT(unit_len);
    CHECK_INPUT(aux_mat);

    return density_map_cuda_forward_naive(node_pos,
                                          node_size,
                                          node_weight,
                                          unit_len,
                                          aux_mat,
                                          num_bin_x,
                                          num_bin_y,
                                          num_bin_z,
                                          num_nodes,
                                          min_node_w,
                                          min_node_h,
                                          margin,
                                          clamp_node);
}

at::Tensor density_map_normalize_node(torch::Tensor node_pos,
                                      torch::Tensor node_size,
                                      torch::Tensor node_weight,
                                      torch::Tensor expand_ratio,
                                      torch::Tensor unit_len,
                                      torch::Tensor normalize_node_info,
                                      int num_bin_x,
                                      int num_bin_y,
                                      int num_bin_z,
                                      int num_nodes) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_size);
    CHECK_INPUT(node_weight);
    CHECK_INPUT(expand_ratio);
    CHECK_INPUT(unit_len);
    CHECK_INPUT(normalize_node_info);

    return density_map_cuda_normalize_node(node_pos,
                                           node_size,
                                           node_weight,
                                           expand_ratio,
                                           unit_len,
                                           normalize_node_info,
                                           num_bin_x,
                                           num_bin_y,
                                           num_bin_z,
                                           num_nodes);
}

torch::Tensor density_backward(torch::Tensor normalize_node_info,
                               torch::Tensor sorted_node_map,
                               torch::Tensor density_map,
                               int num_bin_x,
                               int num_bin_y,
                               int num_bin_z,
                               int num_nodes) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(density_map);

    return density_backward_cuda(
        normalize_node_info, sorted_node_map, density_map, num_bin_x, num_bin_y, num_bin_z, num_nodes);
}

torch::Tensor apply_kernel(torch::Tensor density_map,
                           torch::Tensor aux_mat,
                           int num_bin_x,
                           int num_bin_y,
                           int num_bin_z,
                           torch::Tensor kernel,
                           int kernelWidth) {
    CHECK_INPUT(density_map);
    CHECK_INPUT(aux_mat);
    CHECK_INPUT(kernel);

    return apply_kernel_cuda(density_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, kernel, kernelWidth);
}

}  // namespace GP3D