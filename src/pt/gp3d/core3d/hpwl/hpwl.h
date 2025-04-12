#pragma once

#include <omp.h>

#include "../../placer3d/database3d.h"
#include "../atomic_op.h"
#include "global.h"

namespace GP3D {

namespace wa_wirelength_hpwl {

torch::Tensor nodePosToPinPos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos);

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl(torch::Tensor node_pos,
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

tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_accurate_hpwl(
    torch::Tensor node_pos,
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

void update_rel_cpos(torch::Tensor& pin_rel_cpos,
                     torch::Tensor pin_id2node_id,
                     torch::Tensor ratio_difference,
                     torch::Tensor current_node_slide_state);

void force_remove_overlap(std::vector<int> macro_list, torch::Tensor& node_pos, int num_macros, torch::Tensor node_size);

torch::Tensor masked_scale_hpwl(torch::Tensor node_pos,
                                torch::Tensor pin_id2node_id,
                                torch::Tensor pin_rel_cpos,
                                torch::Tensor hyperedge_list,
                                torch::Tensor hyperedge_list_end,
                                torch::Tensor net_mask,
                                torch::Tensor hpwl_scale);

torch::Tensor get_hpwl(PlaceData& data, torch::Tensor pin_pos);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_ovlp(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    torch::Tensor net_weight,
    torch::Tensor die_info,
    torch::Tensor node_optim_info,
    float gamma);

torch::Tensor masked_scale_hpwl_cross_chip(torch::Tensor node_pos,
                                           torch::Tensor node_die,
                                           torch::Tensor pin_id2node_id,
                                           torch::Tensor pin_rel_cpos,
                                           torch::Tensor hyperedge_list,
                                           torch::Tensor hyperedge_list_end,
                                           torch::Tensor net_mask,
                                           torch::Tensor hpwl_scale);

tuple<torch::Tensor, torch::Tensor> get_hpwl_cross_chip(PlaceData& data,
                                                        torch::Tensor node_pos,
                                                        torch::Tensor node_die);

torch::Tensor place_via_to_optim(torch::Tensor node_pos,
                                 torch::Tensor node_die,
                                 torch::Tensor pin_id2node_id,
                                 torch::Tensor pin_rel_cpos,
                                 torch::Tensor hyperedge_list,
                                 torch::Tensor hyperedge_list_end,
                                 torch::Tensor net_mask);

}  // namespace wa_wirelength_hpwl

}  // namespace GP3D