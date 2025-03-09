#pragma once

#include <omp.h>

#include "../atomic_op.h"
#include "global.h"
#include "placer/database.h"

namespace wa_wirelength_hpwl {

torch::Tensor nodePosToPinPos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl(torch::Tensor node_pos,
                                                                                     torch::Tensor pin_id2node_id,
                                                                                     torch::Tensor pin_rel_cpos,
                                                                                     torch::Tensor node2pin_list,
                                                                                     torch::Tensor node2pin_list_end,
                                                                                     torch::Tensor hyperedge_list,
                                                                                     torch::Tensor hyperedge_list_end,
                                                                                     torch::Tensor net_mask,
                                                                                     float gamma,
                                                                                     bool deterministic = true);

torch::Tensor masked_scale_hpwl(torch::Tensor node_pos,
                                torch::Tensor pin_id2node_id,
                                torch::Tensor pin_rel_cpos,
                                torch::Tensor hyperedge_list,
                                torch::Tensor hyperedge_list_end,
                                torch::Tensor net_mask,
                                torch::Tensor hpwl_scale);

torch::Tensor get_hpwl(PlaceData& data, torch::Tensor pin_pos);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_hpwl_cross_chip(PlaceData& data,
                                                        torch::Tensor node_pos,
                                                        torch::Tensor node_die);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_hpwl_formatted(PlaceData& data,
                                                                      torch::Tensor node_pos,
                                                                      torch::Tensor node_die);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cross_chip(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    float gamma,
    bool deterministic = true);

torch::Tensor masked_scale_cross_chip_hpwl(torch::Tensor node_pos,
                                torch::Tensor node_die,
                                torch::Tensor pin_id2node_id,
                                torch::Tensor pin_rel_cpos,
                                torch::Tensor hyperedge_list,
                                torch::Tensor hyperedge_list_end,
                                torch::Tensor net_mask,
                                torch::Tensor hpwl_scale);

tuple<torch::Tensor, torch::Tensor> place_via_to_optim(torch::Tensor node_pos,
                                 torch::Tensor node_die,
                                 torch::Tensor pin_id2node_id,
                                 torch::Tensor pin_rel_cpos,
                                 torch::Tensor hyperedge_list,
                                 torch::Tensor hyperedge_list_end,
                                 torch::Tensor net_mask);

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_via_from_optim(PlaceData& data,
                                                        torch::Tensor node_pos,
                                                        torch::Tensor node_die);

}  // namespace wa_wirelength_hpwl