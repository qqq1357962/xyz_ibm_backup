
#include "initializer.h"

namespace GP3D {

void init_params(torch::Tensor mov_node_pos,
                 std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn,
                 int mov_lhs,
                 int mov_rhs,
                 torch::Tensor conn_fix_node_pos,
                 //  ElectronicDensityLayer& density_map_layer,
                 vector<ElectronicDensityLayer>& density_map_layers,
                 torch::Tensor mov_node_size,
                 torch::Tensor init_density_map,
                 torch::optim::Optimizer& optimizer,
                 ParamScheduler& ps,
                 NodeData3D& data,
                 torch::Tensor node_die_patoh,
                 torch::Tensor node_slide_state) {
    mov_node_pos = trunc_node_pos_fn(mov_node_pos);
    torch::Tensor conn_node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs), "..."});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);

    // TODO: param scheduler
    torch::Tensor node_weight;
    torch::Tensor overflow;
    torch::Tensor den_loss;

    /* 3 density layers: cell | cell | via */
    vector<torch::Tensor> den_losses(density_map_layers.size());
    for (int i = 0; i < density_map_layers.size(); i++) {
        node_weight = data.mov_node_weights[i];

        auto den_val_list = density_map_layers[i].forward(mov_node_pos, mov_node_size, init_density_map, node_weight, data.macro_mask);
        if (!i) {
            den_loss = den_val_list[0];
            overflow = den_val_list[1];
        } else {
            den_loss += den_val_list[0];
        }
    }

    auto node_die = density_map_layers[0].node_die.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).clone();
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
                                                      ps.wa_coeff,
                                                      data.hpwl_scale,
                                                      density_map_layers[0].ratio_difference,
                                                      data.macro_mask,
                                                      node_die_patoh,
                                                      node_slide_state,
                                                      data.die_info,
                                                      node_die);
    auto wl_loss = wl_val_list[0];
    auto hpwl = wl_val_list[1];

    auto [wl_grad, density_grad] = calc_grad(optimizer, mov_node_pos, wl_loss, den_loss);

    // FIXME: which channel
    double init_density_weight =
        st::setting.force_coeff_2d == 0
            ? (wl_grad.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), 2}).norm(1) /
               density_grad.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), 2}).norm(1))
                  .detach()
                  .item<double>()
            : (wl_grad.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), Slice(0, 2)}).norm(1) /
               density_grad.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), Slice(0, 2)}).norm(1))
                  .detach()
                  .item<double>();

    init_density_weight = ((st::setting.force_coeff_2d == 0) && (st::setting.net_weight_coef == 0)) ? 1 : init_density_weight; // FIXME:
    
    // double init_density_weight =
    //     (wl_grad.index({"...", Slice(2)}).norm(1) / density_grad.index({"...", Slice(2)}).norm(1))
    //         .detach()
    //         .item<double>();

    logger.info("Init density weight %.3E", init_density_weight);

    ps.set_init_param(init_density_weight);
}

double estimate_initial_learning_rate(
    const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor _x_k,
    double lr) {
    // create a new x_k
    torch::Tensor x_k = (_x_k).clone().detach().requires_grad_(true);
    auto [obj_k, g_k] = obj_and_grad_fn(x_k);
    torch::Tensor x_k_1 = ((x_k - lr * g_k)).clone().detach().requires_grad_(true);
    auto [obj_k_1, g_k_1] = obj_and_grad_fn(x_k_1);
    // FIXME:
    // return ((x_k - x_k_1).norm(2) / (g_k - g_k_1).norm(2)).item().toDouble();
    // FIXME
    double numerator = (lr * g_k).norm(2).item<double>();
    double denominator = (g_k - g_k_1).norm(2).item<double>();
    if (denominator <= 0)
        return -1;
    else
        return numerator / denominator;
        // return ((lr * g_k).norm(2) / (g_k - g_k_1).norm(2)).item<double>();
}

}  // namespace GP3D