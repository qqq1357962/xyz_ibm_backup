#pragma once

#include "global.h"
#include "hpwl/hpwl.h"

class WAWirelengthLossAndHPWL : public Function<WAWirelengthLossAndHPWL> {
public:
    static variable_list forward(AutogradContext *ctx,
                                 at::Tensor node_pos,
                                 at::Tensor pin_id2node_id,
                                 at::Tensor pin_rel_cpos,
                                 at::Tensor node2pin_list,
                                 at::Tensor node2pin_list_end,
                                 at::Tensor hyperedge_list,
                                 at::Tensor hyperedge_list_end,
                                 at::Tensor net_mask,
                                 double gamma,
                                 at::Tensor hpwl_scale,
                                 vector<int> macro_list,
                                 at::Tensor node_die = torch::empty({0})) {
        // Save data for backward in context
        torch::Tensor partial_wa_wl, node_grad, partial_hpwl;
        auto partial_cross_wl = torch::zeros_like(node_pos);
        auto partial_one_die_wl = torch::zeros_like(node_pos);
        if (node_die.numel()) {
            // if (false) {
            std::tie(partial_wa_wl, node_grad, partial_hpwl) =
                wa_wirelength_hpwl::merged_forward_backward_with_hpwl_cross_chip(node_pos,
                                                                                 node_die,
                                                                                 pin_id2node_id,
                                                                                 pin_rel_cpos,
                                                                                 hyperedge_list,
                                                                                 hyperedge_list_end,
                                                                                 net_mask,
                                                                                 gamma,
                                                                                 true);
        } else {
            std::tie(partial_wa_wl, node_grad, partial_hpwl, partial_cross_wl, partial_one_die_wl) =
                wa_wirelength_hpwl::merged_forward_backward_with_hpwl(node_pos,
                                                                      pin_id2node_id,
                                                                      pin_rel_cpos,
                                                                      node2pin_list,
                                                                      node2pin_list_end,
                                                                      hyperedge_list,
                                                                      hyperedge_list_end,
                                                                      net_mask,
                                                                      gamma,
                                                                      true);

            // std::cout << "cross_wl" << torch::sum(partial_cross_wl).item<int>() << std::endl;
            // std::cout << "one_die_wl" << torch::sum(partial_one_die_wl).item<int>() << std::endl;
        }

        at::Tensor sum_hpwl = torch::round(partial_hpwl * hpwl_scale).sum();
        for(int i=0;i<macro_list.size();i++)
        {
            node_grad[macro_list[i]]*=st::setting.macro_hpwl_scale;
        }
        ctx->save_for_backward({node_grad});

        // if (st::setting.iteration % 20 == 0) {
        //     printf("=====================\n");
        //     printf("WA WL : %.3E\n", torch::sum(partial_wa_wl).item<float>());
        //     printf("WL : %.3E\n", sum_hpwl.item<float>());
        //     printf("wa coef : %.2f\n", gamma);
        // }

        return {torch::sum(partial_wa_wl), sum_hpwl};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
        // Use data saved in forward
        at::Tensor node_grad = ctx->get_saved_variables()[0];
        at::Tensor wa_grad_out = grad_outputs[0];
        // TODO:
        // cout << (node_grad * wa_grad_out).mean(0) << endl;
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
                Variable()};//sometimes it can parse in "node die" option
    }
};  // END MODULE

//---------------------------------------------------------------------