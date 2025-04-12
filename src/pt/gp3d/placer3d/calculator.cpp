
#include "calculator.h"

namespace GP3D {

torch::Tensor calc_loss(torch::Tensor wl_loss, torch::Tensor density_loss, ParamScheduler& ps) {
    torch::Tensor loss = torch::zeros(wl_loss.sizes(), torch::dtype(torch::kFloat64)).contiguous();
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
    return loss;
}

void apply_precond(torch::Tensor mov_node_pos, ParamScheduler& ps) {
    if (!ps.use_precond) return;
    // FIXME: https://github.com/pytorch/pytorch/pull/40887
    mov_node_pos.mutable_grad() /= ps.precond_weight.to(mov_node_pos.device());
}

// For Nesterov
tuple<torch::Tensor, torch::Tensor, torch::Tensor> calc_obj_and_grad(torch::Tensor mov_node_pos,
                                                      std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                      torch::Tensor mov_node_size,
                                                      torch::Tensor init_density_map,
                                                    //   ElectronicDensityLayer& density_map_layer,
                                                      vector<ElectronicDensityLayer>& density_map_layers,
                                                      torch::Tensor conn_fix_node_pos,
                                                      ParamScheduler& ps,
                                                      NodeData3D& data,
                                                      torch::Tensor node_die_patoh,
                                                      torch::Tensor current_node_slide_state) {
    // we disable merged_forward_backward in C++ version since it is quite complicated
    auto [mov_lhs, mov_rhs] = data.movable_index;
    mov_rhs = data.iopin_mov_lhs;
    mov_node_pos = constraint_fn(mov_node_pos);

    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    if (mov_node_pos.grad().defined()) {
        mov_node_pos.grad().zero_();
    } else {
        mov_node_pos.mutable_grad() = torch::zeros_like(mov_node_pos).detach();
    }

    // TODO:
    torch::Tensor node_weight;
    torch::Tensor den_loss;

    /* 2 density layers: cell | cell | via */
    density_map_layers[0].density_weight_local = ps.density_weight_local.to(mov_node_pos.device());
    vector<torch::Tensor> den_losses(density_map_layers.size());
    for (int i = 0; i < density_map_layers.size(); i++) {
        node_weight = data.mov_node_weights[i];
        auto den_val_list = density_map_layers[i].forward(mov_node_pos, mov_node_size, init_density_map, node_weight, data.macro_mask);
        if (!i) {
            den_loss = den_val_list[0];
        } else {
            den_loss += den_val_list[0];
        }
    }

    auto node_die = 1 - density_map_layers[0].node_die.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).clone();
    node_die = torch::cat({node_die, data.node_die.index({Slice(data.cell_mov_rhs, None)})});
    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos,
                                                      data.pin_id2node_id,
                                                      data.pin_rel_cpos,
                                                      data.pin_rel_cpos_top,
                                                      data.pin_rel_cpos_bot,
                                                      data.node2pin_list,
                                                      data.node2pin_list_end,
                                                      data.hyperedge_list,
                                                      data.hyperedge_list_end,
                                                      data.net_mask,
                                                      data.net_weight * ps.net_weight_coef,
                                                    //   st::setting.net_weight_coef * data.net_weight,
                                                      ps.wa_coeff,
                                                      data.hpwl_scale,
                                                      density_map_layers[0].ratio_difference,
                                                      data.macro_mask,
                                                      node_die_patoh,
                                                      current_node_slide_state,
                                                      data.die_info,
                                                      node_die);

    torch::Tensor loss = calc_loss(wl_val_list[0], den_loss, ps);  // FIXME:

    loss.backward();
    apply_precond(mov_node_pos, ps);
    mov_node_pos.mutable_grad().index({"...", Slice(0, 2)}) *= st::setting.force_coeff_2d;
    // mov_node_pos.index(rhs...)mutable_grad().index({"...", Slice(0, 2)}) *= st::setting.force_coeff_2d;
    // if (data.node_wgt_grad.numel()) mov_node_pos.mutable_grad() *= data.node_wgt_grad.unsqueeze(1).to(mov_node_pos.device());
    
    torch::Tensor grad = mov_node_pos.grad();
    torch::Tensor node_slide_grad = wl_val_list[2];

    return {loss, grad, node_slide_grad};
}

tuple<torch::Tensor, torch::Tensor> calc_grad(torch::optim::Optimizer& optimizer,
                                              torch::Tensor mov_node_pos,
                                              torch::Tensor wl_loss,
                                              torch::Tensor density_loss) {
    optimizer.zero_grad();
    wl_loss.backward({}, c10::optional<bool>(true));
    torch::Tensor wl_grad = mov_node_pos.grad().detach().clone();  // FIXME:
    optimizer.zero_grad();
    density_loss.backward({}, c10::optional<bool>(true));
    torch::Tensor density_grad = mov_node_pos.grad().detach().clone();
    optimizer.zero_grad();
    return {wl_grad, density_grad};
}

}  // namespace GP3D