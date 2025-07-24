#include "evaluator.h"

// Function TODO:
// 1) wa_wirelength_hpwl::get_hpwl
// 2) wa_wirelength_hpwl::nodePosToPinPos
// 3) ElectronicDensityLayer::direct_calc_overflow

tuple<torch::Tensor, torch::Tensor> get_obj_value(torch::Tensor pin_pos, torch::Tensor density_map, PlaceData& data) {
    torch::NoGradGuard no_grad;
    torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach()));
    torch::Tensor overflow_sum = ((density_map - st::setting.target_density) * data.bin_area).clamp_(0.0).sum();
    torch::Tensor overflow = overflow_sum / data.__total_mov_area_without_filler__;
    return {hpwl, overflow};
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor> evaluate_placement(torch::Tensor node_pos,
                                                       ElectronicDensityLayer& density_map_layer,
                                                       torch::Tensor init_density_map,
                                                       PlaceData& data) {
    // NOTE: since some nets are masked in WAWirelengthLossAndHPWL, hpwl
    // from WAWirelengthLossAndHPWL may underestimate, this function return the
    // exact value of hpwl
    // Original overflow calculation uses the clamp node size (expand ratio),
    // this function uses the exact node size to evaluate the overflow
    torch::NoGradGuard no_grad;
    auto [mov_lhs, mov_rhs] = data.cell_movable_index;
    auto [fix_lhs, fix_rhs] = data.fixed_connected_index;
    torch::Tensor conn_node_pos =
        torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})}, 0);
    // conn_node_pos = node_pos;
    torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
        conn_node_pos.to(data.device), data.pin_id2node_id.to(data.device), data.pin_rel_cpos.to(data.device));

    // compute naive density map
    int num_nodes = mov_rhs - mov_lhs;
    // torch::Tensor mov_node_weight = data.node_size.new_ones(num_nodes);
    torch::Tensor mov_node_weight = torch::ones(num_nodes, torch::dtype(torch::kFloat).device(node_pos.device()));
    torch::Tensor aux_mat = init_density_map.clone();
    auto density_map = density_map_forward_naive(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                 data.node_size.to(data.device),
                                                 mov_node_weight.to(data.device),
                                                 data.unit_len.to(data.device),
                                                 aux_mat.to(data.device),
                                                 data.num_bin_x,
                                                 data.num_bin_y,
                                                 num_nodes,
                                                 -1.0,
                                                 -1.0,
                                                 1e-4,
                                                 false);

    return get_obj_value(pin_pos, density_map, data);
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor> fast_evaluator(torch::Tensor mov_node_pos,
                                                   std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                   torch::Tensor mov_node_size,
                                                   torch::Tensor init_density_map,
                                                   ElectronicDensityLayer& density_map_layer,
                                                   torch::Tensor conn_fix_node_pos,
                                                   ParamScheduler& ps,
                                                   PlaceData& data) {
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_node_pos = constraint_fn(mov_node_pos);
    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    torch::Tensor masked_hpwl = wa_wirelength_hpwl::masked_scale_hpwl(conn_node_pos,
                                                                      data.pin_id2node_id,
                                                                      data.pin_rel_cpos,
                                                                      data.hyperedge_list,
                                                                      data.hyperedge_list_end,
                                                                      data.net_mask,
                                                                      data.hpwl_scale);
    auto [overflow, tmp] = density_map_layer.direct_calc_overflow(mov_node_pos, mov_node_size, init_density_map);
    return {masked_hpwl, overflow};
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_wl_cross_chip(torch::Tensor node_pos,
                                                                          torch::Tensor node_die,
                                                                          PlaceData& data) {
    // exact value of hpwl cross two chips
    torch::NoGradGuard no_grad;
    auto [hpwl1, hpwl2, hpwl_ovlp] = wa_wirelength_hpwl::get_hpwl_cross_chip(data, node_pos, node_die);

    // FIXME:
    return {hpwl1.sum() * data.hpwl_scale[0].item<float>(), hpwl2.sum() * data.hpwl_scale[0].item<float>(), hpwl_ovlp.sum() * data.hpwl_scale[0].item<float>()};
}  // END MODULE

int calc_via_cost(torch::Tensor node_die, PlaceData& data) {
    torch::Tensor net_cut_info = torch::zeros({data.num_nets, 2}, torch::dtype(torch::kInt));
    const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = data.pin_id2node_id.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = data.hyperedge_list.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = data.hyperedge_list_end.accessor<int64_t, 1>();
    const torch::TensorAccessor<int, 1> cell_die_at = node_die.accessor<int, 1>();
    for (int i = 0; i != data.num_nets; i++) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end_at[i - 1];
        }
        int64_t end_idx = hyperedge_list_end_at[i];
        if (end_idx != start_idx) {
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list_at[idx];
                int64_t node_id = pin_id2node_id_at[pin_id];
                if (node_id > data.num_nodes) continue;

                int c_id = cell_die_at[node_id];

                if (c_id != 0 && c_id != 1) continue;

                net_cut_info[i][c_id] += 1;
            }
        }
    }
    auto bonding_map = torch::_cast_Int(torch::prod(net_cut_info, 1) != 0);
    int cutsize = (bonding_map).sum().item<int>();
    return cutsize;
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_wl_ovhd_cross_chip(torch::Tensor node_pos,
                                                                                              torch::Tensor node_die,
                                                                                              PlaceData& data) {
    // exact value of hpwl cross two chips
    torch::NoGradGuard no_grad;
    auto [hpwl1, hpwl2, hpwl_ovhd, via_bbox] = wa_wirelength_hpwl::evaluate_via_from_optim(data, node_pos, node_die);

    // FIXME:
    return {hpwl1.sum() * data.hpwl_scale[0],
            hpwl2.sum() * data.hpwl_scale[0],
            hpwl_ovhd.sum() * data.hpwl_scale[0],
            via_bbox * data.hpwl_scale[0]};
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_evaluator_multi_circuit(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_maps,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data) {
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_node_pos = constraint_fn(mov_node_pos);
    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);

    torch::Tensor overflow = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    torch::Tensor masked_hpwl = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    /* 2 circuits: chip_0 | chip_1 */
    torch::Tensor masked_hpwls = torch::zeros({2}, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));

    if (true) {
        for (int i = 0; i < 2; i++) {
            torch::Tensor masked_hpwl_chip = wa_wirelength_hpwl::masked_scale_hpwl(conn_node_pos,
                                                                                   data.pin_id2node_id,
                                                                                   data.pin_rel_cpos,
                                                                                   data.hyperedge_list_cc[i],
                                                                                   data.hyperedge_list_end_cc[i],
                                                                                   data.net_mask,
                                                                                   data.hpwl_scale);
            masked_hpwl += masked_hpwl_chip;
            masked_hpwls[i] = masked_hpwl_chip;
        }
    } else if (st::setting.net_type == "monon") {
        masked_hpwls[0] = wa_wirelength_hpwl::masked_scale_cross_chip_hpwl(conn_node_pos,
                                                                           data.node_die,
                                                                           data.pin_id2node_id,
                                                                           data.pin_rel_cpos,
                                                                           data.hyperedge_list,
                                                                           data.hyperedge_list_end,
                                                                           data.net_mask,
                                                                           data.hpwl_scale);
    } else {
        masked_hpwls[0] = wa_wirelength_hpwl::masked_scale_hpwl(conn_node_pos,
                                                                data.pin_id2node_id,
                                                                data.pin_rel_cpos,
                                                                data.hyperedge_list,
                                                                data.hyperedge_list_end,
                                                                data.net_mask,
                                                                data.hpwl_scale);
    }

    /* 3 density layers: cell | cell | via */
    torch::Tensor overflows = torch::zeros({3}, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    torch::Tensor density_maps = torch::zeros({st::setting.num_den_layer, st::setting.num_bin_x, st::setting.num_bin_y},
                                              torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    
    if (st::setting.skip_2_5d) {
        auto [overflow_chip, density_map] =
            density_map_layers[2].direct_calc_overflow(mov_node_pos, mov_node_size, init_density_maps[2]);
        overflows[2] = overflow_chip;
        density_maps[2] = density_map;
    } else {
        for (int i = 0; i < st::setting.num_den_layer; i++) {
            auto [overflow_chip, density_map] =
                density_map_layers[i].direct_calc_overflow(mov_node_pos, mov_node_size, init_density_maps[i]);
            overflows[i] = overflow_chip;
            density_maps[i] = density_map;
        }
    }
    

    return {masked_hpwls, overflows, density_maps};
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor> evaluate_placement_multi_circuit(torch::Tensor node_pos,
                                                                     vector<ElectronicDensityLayer>& density_map_layers,
                                                                     torch::Tensor init_density_maps,
                                                                     NodeData& data) {
    // NOTE: since some nets are masked in WAWirelengthLossAndHPWL, hpwl
    // from WAWirelengthLossAndHPWL may underestimate, this function return the
    // exact value of hpwl
    // Original overflow calculation uses the clamp node size (expand ratio),
    // this function uses the exact node size to evaluate the overflow
    torch::NoGradGuard no_grad;
    auto [mov_lhs, mov_rhs] = data.cell_movable_index;
    auto [fix_lhs, fix_rhs] = data.fixed_connected_index;
    // torch::Tensor conn_node_pos =
    //     torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})}, 0);
    torch::Tensor conn_node_pos = node_pos;

    auto [hpwl1, hpwl2, hpwl_ovlp] = wa_wirelength_hpwl::get_hpwl_cross_chip(data, conn_node_pos, data.node_die);
    auto hpwl = (hpwl1 + hpwl2).sum();

    torch::Tensor overflows = torch::zeros({3}, torch::dtype(node_pos.dtype()).device(node_pos.device()));
    int num_nodes = mov_rhs - mov_lhs;
    for (int i = 0; i < st::setting.num_den_layer; i++) {
        torch::Tensor aux_mat = init_density_maps[i].clone();
        auto density_map = density_map_forward_naive(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                     data.node_size.to(data.device),
                                                     data.mov_node_weights[i].to(data.device),
                                                     data.unit_len.to(data.device),
                                                     aux_mat.to(data.device),
                                                     data.num_bin_x,
                                                     data.num_bin_y,
                                                     num_nodes,
                                                     -1.0,
                                                     -1.0,
                                                     1e-4,
                                                     false);

        torch::Tensor overflow_sum = ((density_map - 1) * data.bin_area).clamp_(0.0).sum();
        torch::Tensor overflow = overflow_sum / data.mov_cell_areas[i];

        overflows[i] = overflow;
    }

    return {hpwl, overflows};
}  // END MODULE
