
#include "calculator.h"

torch::Tensor calc_loss(torch::Tensor wl_loss, torch::Tensor density_loss, ParamScheduler& ps) {
    torch::Tensor loss;
    if (st::setting.quad_penalty) {
        ps.init_density = density_loss.clone();
        ps.density_weight_grad_precond =
            ps.init_density.masked_scatter(ps.init_density > 0, 1 / ps.init_density.index({ps.init_density > 0}));
        ps.quad_penalty_coeff = ps.density_quad_coeff / 2 * ps.density_weight_grad_precond;
        density_loss = density_loss * (1 + ps.quad_penalty_coeff * density_loss);
    }
    if (st::setting.loss_type == "weighted_sum") {
        loss = (wl_loss + ps.density_weight * density_loss) / (1 + ps.density_weight);
    } else if (st::setting.loss_type == "direct") {
        loss = wl_loss + ps.density_weight * density_loss;
    } else if (st::setting.loss_type == "den_only") {
        loss = density_loss;
    } else if (st::setting.loss_type == "wl_only") {
        loss = wl_loss;
    } else
        logger.error("Loss type not defined");
    // logger.info("density_loss: %f, wl_loss: %f", density_loss.item<float>(), wl_loss.item<float>());
    return loss;
}

void apply_precond(torch::Tensor mov_node_pos, ParamScheduler& ps) {
    if (!ps.use_precond) return;
    // FIXME: https://github.com/pytorch/pytorch/pull/40887
    mov_node_pos.mutable_grad() /= ps.precond_weight.to(mov_node_pos.device());
}

// For Nesterov
tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad(torch::Tensor mov_node_pos,
                                                      std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                      torch::Tensor mov_node_size,
                                                      torch::Tensor init_density_map,
                                                      ElectronicDensityLayer& density_map_layer,
                                                      torch::Tensor conn_fix_node_pos,
                                                      ParamScheduler& ps,
                                                      PlaceData& data) {
    // we disable merged_forward_backward in C++ version since it is quite complicated
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_node_pos = constraint_fn(mov_node_pos);
    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    if (mov_node_pos.grad().defined()) {
        mov_node_pos.grad().zero_();
    } else {
        mov_node_pos.mutable_grad() = torch::zeros_like(mov_node_pos).detach();
    }
    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                      data.pin_id2node_id,
                                                      data.pin_rel_cpos,
                                                      data.node2pin_list,
                                                      data.node2pin_list_end,
                                                      data.hyperedge_list,
                                                      data.hyperedge_list_end,
                                                      data.net_mask,
                                                      ps.wa_coeff,
                                                      data.hpwl_scale,
                                                      data.macro_list);
    torch::Tensor node_weight;
    auto den_val_list =
        density_map_layer.forward(mov_node_pos, mov_node_size, init_density_map, data.mov_node_weight, false);
    torch::Tensor loss = calc_loss(wl_val_list[0], den_val_list[0], ps);
    loss.backward();
    apply_precond(mov_node_pos, ps);
    torch::Tensor grad = mov_node_pos.grad();
    return {loss, grad};
}

// For Nesterov
tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad_with_via(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_maps,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data,
    int& c_id) {
    // we disable merged_forward_backward in C++ version since it is quite complicated
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_node_pos = constraint_fn(mov_node_pos);
    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    if (mov_node_pos.grad().defined()) {
        mov_node_pos.grad().zero_();
    } else {
        mov_node_pos.mutable_grad() = torch::zeros_like(mov_node_pos).detach();
    }

    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                      data.pin_id2node_id,
                                                      data.pin_rel_cpos,
                                                      data.node2pin_list,//@@WHY CC
                                                      data.node2pin_list_end,
                                                      data.hyperedge_list_cc[c_id],
                                                      data.hyperedge_list_end_cc[c_id],
                                                      data.net_mask,
                                                      ps.wa_coeff,
                                                      data.hpwl_scale,
                                                      data.macro_list);

    torch::Tensor node_weight = data.mov_node_weights[c_id];
    auto den_val_list =
        density_map_layers[c_id].forward(mov_node_pos, mov_node_size, init_density_maps[c_id], node_weight, false);

    torch::Tensor loss = calc_loss(wl_val_list[0], den_val_list[0], ps);

    loss.backward();
    apply_precond(mov_node_pos, ps);
    torch::Tensor grad = mov_node_pos.grad();
    return {loss, grad};
}

// For Adam
tuple<torch::Tensor, torch::Tensor> calc_grad(torch::optim::Optimizer& optimizer,
                                              torch::Tensor mov_node_pos,
                                              torch::Tensor wl_loss,
                                              torch::Tensor density_loss) {
    optimizer.zero_grad();
    wl_loss.backward({}, c10::optional<bool>(true));
    torch::Tensor wl_grad = mov_node_pos.grad().detach().clone();
    optimizer.zero_grad();
    density_loss.backward({}, c10::optional<bool>(true));
    torch::Tensor density_grad = mov_node_pos.grad().detach().clone();
    optimizer.zero_grad();
    return {wl_grad, density_grad};
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_optimization(
    torch::Tensor mov_node_pos,
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
    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                      data.pin_id2node_id,
                                                      data.pin_rel_cpos,
                                                      data.node2pin_list,
                                                      data.node2pin_list_end,
                                                      data.hyperedge_list,
                                                      data.hyperedge_list_end,
                                                      data.net_mask,
                                                      // FIXME:
                                                      ps.wa_coeff,
                                                      data.hpwl_scale,
                                                      data.macro_list);
    torch::Tensor wl_loss = wl_val_list[0];
    torch::Tensor hpwl = wl_val_list[1];
    torch::Tensor node_weight;
    auto den_val_list = density_map_layer.forward(mov_node_pos, mov_node_size, init_density_map, node_weight, true);
    torch::Tensor density_loss = den_val_list[0];
    torch::Tensor overflow = den_val_list[1];

    torch::Tensor loss = calc_loss(wl_loss, density_loss, ps);
    loss.backward();
    apply_precond(mov_node_pos, ps);
    // calculate objective (hpwl, overflow)
    return {hpwl.detach(), overflow.detach(), mov_node_pos};
}  // END MODULE

//---------------------------------------------------------------------

// For contest multi chip&circuit
tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad_multi_circuit(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_maps,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data,
    PlaceData& via_data) {
    // we disable merged_forward_backward in C++ version since it is quite complicated
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_node_pos = constraint_fn(mov_node_pos);
    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    if (mov_node_pos.grad().defined()) {
        mov_node_pos.grad().zero_();
    } else {
        mov_node_pos.mutable_grad() = torch::zeros_like(mov_node_pos).detach();
    }

    vector<torch::Tensor> wl_losses(2);
    vector<torch::Tensor> den_losses(3);
    torch::Tensor wl_loss;
    torch::Tensor den_loss;
    if (st::setting.net_type == "monon") {
        // auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
        //                                                 data.pin_id2node_id,
        //                                                 data.pin_rel_cpos,
        //                                                 data.hyperedge_list,
        //                                                 data.hyperedge_list_end,
        //                                                 data.net_mask,
        //                                                 ps.wa_coeff,
        //                                                 data.hpwl_scale,
        //                                                 data.node_die);
        // wl_loss = wl_val_list[0];

        /* 2 circuits: chip_0 | chip_1 */
        for (int i = 0; i < 2; i++) {
            auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                            data.pin_id2node_id,
                                                            data.pin_rel_cpos,
                                                            data.node2pin_list,
                                                            data.node2pin_list_end,
                                                            data.hyperedge_list_cc[i],
                                                            data.hyperedge_list_end_cc[i],
                                                            data.net_mask,
                                                            ps.wa_coeffs[i],
                                                            data.hpwl_scale,
                                                            data.macro_list);
            wl_losses[i] = wl_val_list[0];
        }
        wl_loss = wl_losses[0] + wl_losses[1];
    } else {
        auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                        data.pin_id2node_id,
                                                        data.pin_rel_cpos,
                                                        data.node2pin_list,
                                                        data.node2pin_list_end,
                                                        data.hyperedge_list,
                                                        data.hyperedge_list_end,
                                                        data.net_mask,
                                                        ps.wa_coeff,
                                                        data.hpwl_scale,
                                                        data.macro_list);
        wl_loss = wl_val_list[0];
    }

    /* 3 density layers: cell | cell | via */
    // for (int i = 0; i < 2; i++) {  // TODO:
    if (st::setting.skip_2_5d) {
        torch::Tensor node_weight = data.mov_node_weights[2];
        auto den_val_list =
            density_map_layers[2].forward(mov_node_pos, mov_node_size, init_density_maps[2], node_weight, false);
        den_loss = den_val_list[0];
    } else {
        for (int i = 0; i < st::setting.num_den_layer; i++) {  // TODO:
            torch::Tensor node_weight = data.mov_node_weights[i];
            auto den_val_list =
                density_map_layers[i].forward(mov_node_pos, mov_node_size, init_density_maps[i], node_weight, false);
            den_losses[i] = den_val_list[0];
        }
        den_loss = st::setting.num_den_layer == 3 ? (den_losses[0] + den_losses[1] + den_losses[2])
                                                  : (den_losses[0] + den_losses[1]);
    }
    

    torch::Tensor loss = calc_loss(wl_loss, den_loss, ps);
    loss.backward();
    apply_precond(mov_node_pos, ps);

    torch::Tensor grad = mov_node_pos.grad();

    return {loss, grad};
}  // END MODULE
