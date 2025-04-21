#pragma once

#include "global.h"
#include "hpwl/hpwl.h"

namespace GP3D {

class WAWirelengthLossAndHPWL : public Function<WAWirelengthLossAndHPWL> {
public:
    static variable_list forward(AutogradContext *ctx,
                                 at::Tensor node_pos,
                                 at::Tensor pin_id2node_id,
                                 at::Tensor pin_rel_cpos,
                                 at::Tensor pin_rel_cpos_top,
                                 at::Tensor pin_rel_cpos_bot,
                                 at::Tensor node2pin_list,
                                 at::Tensor node2pin_list_end,
                                 at::Tensor hyperedge_list,
                                 at::Tensor hyperedge_list_end,
                                 at::Tensor net_mask,
                                 at::Tensor net_weight,
                                 double gamma,
                                 at::Tensor hpwl_scale,
                                 at::Tensor ratio_difference,
                                 at::Tensor macro_mask,
                                 at::Tensor node_die_patoh,
                                 at::Tensor current_node_slide_state,
                                 at::Tensor die_info = torch::empty({0}),
                                 at::Tensor node_die = torch::empty({0})) {
        // Save data for backward in context
        int cell_count = ratio_difference.size(0);
        auto node_pos_channel = node_pos.index({Slice(0, cell_count), 2});
        // auto pin_rel_cpos_real = pin_rel_cpos_top.clone();
        pin_rel_cpos = pin_rel_cpos_top.clone();//incorrect
        auto ratio = node_pos_channel / die_info[2 * 2 + 1];
        ratio = (-(1.5 + 2 * macro_mask) + ratio * (4 + macro_mask * 4)).unsqueeze(1);
        ratio = ratio.clamp(0,1);
        auto pin_rel_cpos_difference = (pin_rel_cpos_bot + 1e-3) / (pin_rel_cpos_top + 1e-3);
        auto pin_ratio = ratio.index_select(0, pin_id2node_id);
        auto pin_broadcasted_ratio = pin_ratio.expand_as(pin_rel_cpos_difference);
        pin_rel_cpos_difference = (pin_rel_cpos_difference*(1-pin_broadcasted_ratio)+pin_broadcasted_ratio);
        // mov_node_size = mov_node_size_top * ratio + mov_node_size_bot * (1-ratio);
        // pin_rel_cpos_real.slice(0, 0, cell_count).slice(1,0,2) *= (ratio_difference*(1-ratio)+ratio);
        auto ratio_difference_area = ratio_difference.prod(1).sqrt();
        ratio = ratio.squeeze(1);
        auto ratio_multiply = (ratio_difference_area - 1) * 0.39;
        // cout << ratio_multiply.sizes() << endl;
        ratio_multiply = (node_die_patoh - (node_die_patoh.max() + node_die_patoh.min()) / 2).to(ratio_difference_area.options());
        // cout << ratio_multiply << endl;
        
        wa_wirelength_hpwl::update_rel_cpos(pin_rel_cpos, pin_id2node_id, pin_rel_cpos_difference, current_node_slide_state);

        auto net_weight_naive = torch::ones({net_weight.size(0)}, dtype(torch::kFloat)).to(net_weight.device());
        if (st::setting.wa_z_model == "xy") {
            net_weight_naive = net_weight;
        }
        auto [partial_wa_wl, node_grad, partial_hpwl, node_slide_grad, node_orient_grad] =
            wa_wirelength_hpwl::merged_forward_backward_with_hpwl(node_pos,
                                                                  node_die,
                                                                  pin_id2node_id,
                                                                  pin_rel_cpos,
                                                                  node2pin_list,
                                                                  node2pin_list_end,
                                                                //   pin_rel_cpos_real,
                                                                  hyperedge_list,
                                                                  hyperedge_list_end,
                                                                  net_mask,
                                                                  net_weight_naive,
                                                                  macro_mask,
                                                                  ratio_multiply,
                                                                  gamma);
        // auto [partial_wa_wl, node_grad, partial_hpwl] =
        //     wa_wirelength_hpwl::merged_forward_backward_with_accurate_hpwl(node_pos,
        //                                                           node_die,
        //                                                           pin_id2node_id,
        //                                                           pin_rel_cpos,
        //                                                           pin_rel_cpos_top,
        //                                                           pin_rel_cpos_bot,
        //                                                           node2pin_list,
        //                                                           node2pin_list_end,
        //                                                           hyperedge_list,
        //                                                           hyperedge_list_end,
        //                                                           net_mask,
        //                                                           net_weight_naive,
        //                                                           gamma);


        // torch::Tensor node_optim_info = torch::zeros_like(node_die, dtype(torch::kInt));
        // auto [partial_wa_wl1, node_grad1, partial_hpwl1] =
        //     wa_wirelength_hpwl::merged_forward_backward_with_hpwl_ovlp(node_pos,
        //                                                                node_die,
        //                                                                pin_id2node_id,
        //                                                                pin_rel_cpos,
        //                                                             //    pin_rel_cpos_real,
        //                                                                hyperedge_list,
        //                                                                hyperedge_list_end,
        //                                                                net_mask,
        //                                                                net_weight,
        //                                                                die_info,
        //                                                                node_optim_info,
        //                                                                gamma);

        partial_wa_wl.index({"...", 2}) *= 0;
        partial_hpwl.index({"...", 2}) *= 0;



        // if (st::setting.wa_z_model == "z") {
        //     // auto wa_z_norm_coef =
        //     //     (node_grad1.index({"...", Slice(0, 2)}).norm(1) / node_grad1.index({"...", 2}).norm(1))
        //     //         .clamp(INT_MIN, 1);

        //     // node_grad1.index({"...", 2}) *= wa_z_norm_coef * st::setting.wa_coeff_wa_z;
        //     // node_grad1.index({"...", Slice(0, 2)}) *= 1;

        //     // node_grad.index({"...", 2}) *= st::setting.wa_coeff_wa_xy;
        //     // node_grad.index({"...", Slice(0, 2)}) *= 0;

        //     // node_grad = node_grad + node_grad1;

        //     node_grad = node_grad1;
        // }

        // FIXME: grad z
        at::Tensor sum_hpwl = torch::round(partial_hpwl * hpwl_scale).sum();
        ctx->save_for_backward({node_grad});

        // auto non_zero_indices = torch::nonzero(node_slide_grad);
        // cout << node_slide_grad[non_zero_indices[0].item<int>()].item<float>() << endl;

        // if ((st::setting.iteration - 1) % 100 == 0) {
        //     logger.info("==========%d=========", st::setting.iteration);
        //     logger.info("OVLP WL : %.3E", torch::sum(partial_hpwl1 * hpwl_scale).item<float>());
        //     logger.info("Sigmoid WL : %.3E", torch::sum(partial_wa_wl1).item<float>());
        //     logger.info("z pos : %.2f/ (%.2f, %.2f)",
        //                 (node_pos.index({"...", 2}).mean() / die_info[5]).item<float>(),
        //                 node_pos.index({"...", 2}).min().item<float>(),
        //                 node_pos.index({"...", 2}).max().item<float>());
        //     cout << gamma << endl;

        //     // logger.info("%d/%d cells are optimal", node_optim_info.sum().item<int>(), node_optim_info.size(0));
        // }

        return {torch::sum(partial_wa_wl), sum_hpwl, node_slide_grad};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
        // Use data saved in forward
        at::Tensor node_grad = ctx->get_saved_variables()[0];
        at::Tensor wa_grad_out = grad_outputs[0];
        // TODO:

        // node_grad.index({"...", 2}) *= st::setting.iteration == 0 ? 1 : 1;
        // node_grad.index({"...", Slice(0, 2)}) *= st::setting.iteration == 0 ? 1 : st::setting.force_coeff_2d;  // FIXME:

        return {node_grad * wa_grad_out,
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable()};
    }
};  // END MODULE

//---------------------------------------------------------------------

}  // namespace GP3D