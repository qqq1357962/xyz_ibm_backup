#include <torch/torch.h>

torch::Tensor node_pos_to_pin_pos_cuda_forward(torch::Tensor node_pos,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cuda(
    torch::Tensor node_pos,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor node2pin_list,
    torch::Tensor node2pin_list_end,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    float gamma,
    bool deterministic = true);

torch::Tensor masked_scale_hpwl_sum_cuda(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale);

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hpwl_cross_chip_cuda(torch::Tensor node_pos,
                                                                             torch::Tensor node_die,
                                                                             torch::Tensor pin_id2node_id,
                                                                             torch::Tensor pin_rel_cpos,
                                                                             torch::Tensor hyperedge_list,
                                                                             torch::Tensor hyperedge_list_end);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hpwl_formatted_cuda(torch::Tensor node_pos,
                                                                            torch::Tensor node_die,
                                                                            torch::Tensor pin_id2node_id,
                                                                            torch::Tensor pin_rel_cpos,
                                                                            torch::Tensor hyperedge_list,
                                                                            torch::Tensor hyperedge_list_end);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cross_chip_cuda(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    float gamma,
    bool deterministic = true);

torch::Tensor masked_scale_hpwl_sum_cross_chip_cuda(torch::Tensor node_pos,
                                         torch::Tensor node_die,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale);

std::tuple<torch::Tensor, torch::Tensor> place_via_to_optim_cuda(torch::Tensor node_pos,
                                      torch::Tensor node_die,
                                      torch::Tensor pin_id2node_id,
                                      torch::Tensor pin_rel_cpos,
                                      torch::Tensor hyperedge_list,
                                      torch::Tensor hyperedge_list_end,
                                      torch::Tensor net_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_via_from_optim_cuda(torch::Tensor node_pos,
                                                                    torch::Tensor node_die,
                                                                    torch::Tensor pin_id2node_id,
                                                                    torch::Tensor pin_rel_cpos,
                                                                    torch::Tensor hyperedge_list,
                                                                    torch::Tensor hyperedge_list_end);