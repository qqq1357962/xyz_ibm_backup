#include "hpwl_cuda.cuh"

#include "hpwl.h"

namespace GP3D {

namespace wa_wirelength_hpwl {

torch::Tensor nodePosToPinPos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);

    return node_pos_to_pin_pos_cuda_forward(node_pos, pin_id2node_id, pin_rel_cpos);
}

void force_remove_overlap(std::vector<int> macro_list, torch::Tensor& node_pos, int num_nodes, torch::Tensor node_size)
{
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_size);

    return force_remove_overlap_cuda(macro_list, node_pos, num_nodes, node_size);
}

void update_rel_cpos(torch::Tensor& pin_rel_cpos,
                     torch::Tensor pin_id2node_id,
                     torch::Tensor ratio_difference,
                     torch::Tensor current_node_slide_state) {
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(ratio_difference);
    CHECK_INPUT(current_node_slide_state);

    return update_rel_cpos_cuda(
        pin_rel_cpos, pin_id2node_id, ratio_difference, current_node_slide_state);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> merged_forward_backward_with_hpwl(torch::Tensor node_pos,
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
                                                                                     float gamma) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(net_weight);

    return merged_forward_backward_with_hpwl_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, 
            node2pin_list, node2pin_list_end,hyperedge_list, hyperedge_list_end, net_mask, net_weight, macro_mask, ratio_multiply, gamma);
}

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
    float gamma) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(net_weight);

    return merged_forward_backward_with_accurate_hpwl_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, pin_rel_cpos_top, pin_rel_cpos_bot, node2pin_list, node2pin_list_end, hyperedge_list, hyperedge_list_end, net_mask, net_weight, gamma);
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
    torch::Tensor hyperedge_list = data.hyperedge_list;
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end;

    CHECK_INPUT(pin_pos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    return hpwl_cuda(pin_pos, hyperedge_list, hyperedge_list_end);
}

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
    float gamma) {
    torch::Tensor node_pos_norm_z = node_pos.clone();

    // cout << node_pos_norm_z << endl;
    // cout << grad_z_offset << endl;
    // cout << grad_z_scale << endl;

    torch::Tensor grad_z_scale = die_info[5] / 2;
    torch::Tensor grad_z_offset = die_info[5] / 4;

    node_pos_norm_z.index({"...", Slice(2)}) -= grad_z_offset;
    node_pos_norm_z.index({"...", Slice(2)}) /= grad_z_scale;
    // node_pos_norm_z.index({"...", Slice(2)}).clamp_(1e-4);
    // TODO:

    // cout << node_pos_norm_z << endl;

    CHECK_INPUT(node_pos_norm_z);
    CHECK_INPUT(node_die);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(net_weight);

    return merged_forward_backward_with_hpwl_ovlp_cuda(node_pos_norm_z,
                                                       node_die,
                                                       die_info,
                                                       pin_id2node_id,
                                                       pin_rel_cpos,
                                                       hyperedge_list,
                                                       hyperedge_list_end,
                                                       net_mask,
                                                       net_weight,
                                                       node_optim_info,
                                                       gamma,
                                                       st::setting.correlate_bbox,
                                                       st::setting.bbox_correlation);
}

torch::Tensor masked_scale_hpwl_cross_chip(torch::Tensor node_pos,
                                           torch::Tensor node_die,
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
    return masked_scale_hpwl_cross_chip_cuda(
        node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale);
}

tuple<torch::Tensor, torch::Tensor> get_hpwl_cross_chip(PlaceData& data,
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

    auto [wl1, wl2] =
        hpwl_cross_chip_cuda(node_pos, node_die, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end);

    return {wl1.sum(), wl2.sum()};
}

torch::Tensor place_via_to_optim(torch::Tensor node_pos,
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

}  // namespace wa_wirelength_hpwl

}  // namespace GP3D