#include "density_smooth_cuda.cuh"

#include "density_smooth.h"


torch::Tensor density_smooth_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_bin_z,
                                  int num_nodes) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(aux_mat);


    return density_smooth_cuda_forward(
        normalize_node_info, sorted_node_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, num_nodes);
}

torch::Tensor density_smooth_backward(torch::Tensor normalize_node_info,
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

    return density_smooth_cuda_backward(
        normalize_node_info, grad_mat, sorted_node_map, node_grad, grad_weight, num_bin_x, num_bin_y, num_bin_z, num_nodes);
}

at::Tensor density_smooth_normalize_node(torch::Tensor node_pos,
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

    return density_smooth_cuda_normalize_node(
        node_pos, node_size, node_weight, expand_ratio, unit_len, normalize_node_info, num_bin_x, num_bin_y, num_bin_z, num_nodes);
}