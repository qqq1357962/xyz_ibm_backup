#include <torch/torch.h>

namespace GP3D {

torch::Tensor node_pos_to_pin_pos_cuda_forward(torch::Tensor node_pos,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cuda(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor node2pin_list,
    torch::Tensor node2pin_list_end,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    torch::Tensor net_weight,
    torch::Tensor macro_mask,
    torch::Tensor ratio_multiply,
    float gamma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
merged_forward_backward_overlay_with_hpwl_cuda(torch::Tensor node_pos,
                                               torch::Tensor node_die,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos,
                                               torch::Tensor node2pin_list,
                                               torch::Tensor node2pin_list_end,
                                               torch::Tensor hyperedge_list,
                                               torch::Tensor hyperedge_list_end,
                                               torch::Tensor current_node_rotate_state,
                                               torch::Tensor net_mask,
                                               torch::Tensor net_weight,
                                               torch::Tensor macro_mask,
                                               torch::Tensor ratio_multiply,
                                               torch::Tensor direction,
                                               float gamma);

void update_rel_cpos_cuda(torch::Tensor& pin_rel_cpos,
                     torch::Tensor pin_id2node_id,
                     torch::Tensor ratio_difference,
                     torch::Tensor current_node_slide_state);

void update_rel_cpos_rotate_cuda(torch::Tensor& pin_rel_cpos,
                                 torch::Tensor pin_id2node_id,
                                 torch::Tensor ratio_difference,
                                 torch::Tensor current_node_slide_state,
                                 torch::Tensor current_node_rotate_state,
                                 torch::Tensor rotate_direction);

void update_rel_cpos_overlay_cuda(torch::Tensor& pin_rel_cpos,
                                  torch::Tensor pin_id2node_id,
                                  torch::Tensor ratio_difference,
                                  torch::Tensor current_node_slide_state,
                                  torch::Tensor macro_mask,
                                  torch::Tensor& direction,
                                  bool rotate_90);

torch::Tensor masked_scale_hpwl_sum_cuda(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale,
                                         torch::Tensor node_orient_state);

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end);


void force_remove_overlap_cuda(std::vector<int> macro_list, torch::Tensor& node_pos, int num_nodes, torch::Tensor node_size);

torch::Tensor masked_scale_hpwl_cross_chip_cuda(torch::Tensor node_pos,
                                                torch::Tensor node_die,
                                                torch::Tensor pin_id2node_id,
                                                torch::Tensor pin_rel_cpos,
                                                torch::Tensor hyperedge_list,
                                                torch::Tensor hyperedge_list_end,
                                                torch::Tensor net_mask,
                                                torch::Tensor hpwl_scale);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_accurate_hpwl_cuda(torch::Tensor node_pos,
                                                               torch::Tensor node_die,
                                                               torch::Tensor pin_id2node_id,
                                                               torch::Tensor pin_rel_cpos,
                                                               torch::Tensor pin_rel_cpos_top,
                                                               torch::Tensor pin_rel_cpos_bot,
                                                               torch::Tensor node2pin_list,
                                                               torch::Tensor node2pin_list_end,
                                                               torch::Tensor hyperedge_list,
                                                               torch::Tensor hyperedge_list_end,
                                                               torch::Tensor net_mask,
                                                               torch::Tensor net_weight,
                                                               float gamma);


std::tuple<torch::Tensor, torch::Tensor> hpwl_cross_chip_cuda(torch::Tensor node_pos,
                                                              torch::Tensor node_die,
                                                              torch::Tensor pin_id2node_id,
                                                              torch::Tensor pin_rel_cpos,
                                                              torch::Tensor hyperedge_list,
                                                              torch::Tensor hyperedge_list_end);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_ovlp_cuda(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor die_info,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    torch::Tensor net_weight,
    torch::Tensor node_optim_info,
    float gamma, bool correlate_bbox, float bbox_correlation);

torch::Tensor place_via_to_optim_cuda(torch::Tensor node_pos,
                                      torch::Tensor node_die,
                                      torch::Tensor pin_id2node_id,
                                      torch::Tensor pin_rel_cpos,
                                      torch::Tensor hyperedge_list,
                                      torch::Tensor hyperedge_list_end,
                                      torch::Tensor net_mask);
}  // namespace GP3D