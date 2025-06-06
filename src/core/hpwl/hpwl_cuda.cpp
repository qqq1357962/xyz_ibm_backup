#include "hpwl_cuda.cuh"

#include "hpwl.h"

namespace wa_wirelength_hpwl {

torch::Tensor nodePosToPinPos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);

    return node_pos_to_pin_pos_cuda_forward(node_pos, pin_id2node_id, pin_rel_cpos);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl(torch::Tensor node_pos,
                                                                                     torch::Tensor pin_id2node_id,
                                                                                     torch::Tensor pin_rel_cpos,
                                                                                     torch::Tensor node2pin_list,
                                                                                     torch::Tensor node2pin_list_end,
                                                                                     torch::Tensor hyperedge_list,
                                                                                     torch::Tensor hyperedge_list_end,
                                                                                     torch::Tensor net_mask,
                                                                                     float gamma,
                                                                                     bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(node2pin_list);
    CHECK_INPUT(node2pin_list_end);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);

    return merged_forward_backward_with_hpwl_cuda(
        node_pos, pin_id2node_id, pin_rel_cpos, node2pin_list, node2pin_list_end, hyperedge_list, hyperedge_list_end, net_mask, gamma, deterministic);
}

torch::Tensor masked_scale_hpwl(torch::Tensor node_pos,
                                torch::Tensor pin_id2node_id,
                                torch::Tensor pin_rel_cpos,
                                torch::Tensor hyperedge_list,
                                torch::Tensor hyperedge_list_end,
                                torch::Tensor net_mask,
                                torch::Tensor hpwl_scale) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(hpwl_scale);
    return masked_scale_hpwl_sum_cuda(
        node_pos, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale);
}

torch::Tensor get_hpwl(PlaceData& data, torch::Tensor pin_pos) {
    torch::Tensor hyperedge_list = data.hyperedge_list.to(pin_pos.device());
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end.to(pin_pos.device());

    CHECK_INPUT(pin_pos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    return hpwl_cuda(pin_pos, hyperedge_list, hyperedge_list_end);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_hpwl_cross_chip(PlaceData& data,
                                                        torch::Tensor node_pos,
                                                        torch::Tensor node_die) {
    torch::Tensor pin_id2node_id = data.pin_id2node_id.to(node_pos.device());//@@@
    torch::Tensor hyperedge_list = data.hyperedge_list.to(node_pos.device());
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end.to(node_pos.device());
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos.to(node_pos.device());

    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(pin_rel_cpos);

    auto [wl1, wl2, wl_ovlp] =
        hpwl_cross_chip_cuda(node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end);

    return {wl1, wl2, wl_ovlp};
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_hpwl_formatted(PlaceData& data,
                                                                      torch::Tensor node_pos,
                                                                      torch::Tensor node_die) {
    // TODO:
    torch::Tensor pin_id2node_id = data.pin_id2node_id.to(node_pos.device());
    torch::Tensor hyperedge_list = data.hyperedge_list.to(node_pos.device());
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end.to(node_pos.device());
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos.to(node_pos.device());

    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(pin_rel_cpos);

    return hpwl_formatted_cuda(node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl_cross_chip(
    torch::Tensor node_pos,
    torch::Tensor node_die,
    torch::Tensor pin_id2node_id,
    torch::Tensor pin_rel_cpos,
    torch::Tensor hyperedge_list,
    torch::Tensor hyperedge_list_end,
    torch::Tensor net_mask,
    float gamma,
    bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);

    return merged_forward_backward_with_hpwl_cross_chip_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, gamma, deterministic);
}

torch::Tensor masked_scale_cross_chip_hpwl(torch::Tensor node_pos,
                                torch::Tensor node_die,
                                torch::Tensor pin_id2node_id,
                                torch::Tensor pin_rel_cpos,
                                torch::Tensor hyperedge_list,
                                torch::Tensor hyperedge_list_end,
                                torch::Tensor net_mask,
                                torch::Tensor hpwl_scale) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(hpwl_scale);
    return masked_scale_hpwl_sum_cross_chip_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale);
}

tuple<torch::Tensor, torch::Tensor> place_via_to_optim(torch::Tensor node_pos,
                                 torch::Tensor node_die,
                                 torch::Tensor pin_id2node_id,
                                 torch::Tensor pin_rel_cpos,
                                 torch::Tensor hyperedge_list,
                                 torch::Tensor hyperedge_list_end,
                                 torch::Tensor net_mask) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(pin_rel_cpos);

    return place_via_to_optim_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_via_from_optim(PlaceData& data,
                                                        torch::Tensor node_pos,
                                                        torch::Tensor node_die) {
    torch::Tensor pin_id2node_id = data.pin_id2node_id.to(node_pos.device());
    torch::Tensor hyperedge_list = data.hyperedge_list.to(node_pos.device());
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end.to(node_pos.device());
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos.to(node_pos.device());

    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(pin_rel_cpos);

    auto [wl1, wl2, wl_ovlp, via_bbox] =
        evaluate_via_from_optim_cuda(node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end);

    return {wl1.sum(1), wl2.sum(1), wl_ovlp.sum(1), via_bbox};
}

}  // namespace wa_wirelength_hpwl