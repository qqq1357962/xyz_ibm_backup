#pragma once

#include "dct.h"
#include "density_map/density_map.h"
#include "density_smooth/density_smooth.h"
#include "global.h"
#include "placer/visualization.h"

using namespace torch::autograd;

namespace GP3D {

class ElectronicDensityFunction : public Function<ElectronicDensityFunction> {
public:
    static variable_list forward(
        AutogradContext *ctx,
        at::Tensor node_pos,
        at::Tensor node_size,
        at::Tensor node_weight,
        at::Tensor density_weight_local,
        at::Tensor expand_ratio,
        at::Tensor unit_len,
        at::Tensor init_density_map,
        int num_bin_x,
        int num_bin_y,
        int num_bin_z,
        int num_nodes,
        tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale,
        tuple<int, int, std::function<at::Tensor(at::Tensor)>> overflow_helper,
        tuple<at::Tensor, at::Tensor, at::Tensor> sorted_maps,
        bool calc_overflow,
        at::Tensor macro_mask) {
        // Save data for backward in context
        ctx->saved_data["num_bin_x"] = num_bin_x;
        ctx->saved_data["num_bin_y"] = num_bin_y;
        ctx->saved_data["num_bin_z"] = num_bin_z;
        ctx->saved_data["num_nodes"] = num_nodes;

        auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;

        ctx->saved_data["mov_lhs"] = mov_lhs;
        ctx->saved_data["mov_rhs"] = mov_rhs;

        auto [mov_sorted_map, mov_conn_sorted_map, filler_sorted_map] = sorted_maps;

        at::Tensor density_map;
        at::Tensor overflow;
        at::Tensor normalize_node_info;
        // 1) Compute Density Map
        normalize_node_info = node_size.new_empty({num_nodes, 7});  // x_l, x_h, y_l, y_h, z_l, z_h, weight TODO:
        normalize_node_info = density_map_normalize_node(node_pos,
                                                         node_size,
                                                         node_weight,
                                                         expand_ratio,
                                                         unit_len,
                                                         normalize_node_info,
                                                         num_bin_x,
                                                         num_bin_y,
                                                         num_bin_z,
                                                         num_nodes);
        if (calc_overflow) {
            at::Tensor aux_mat = init_density_map.clone();
            at::Tensor mov_density_map = density_map_forward(normalize_node_info,
                                                             mov_conn_sorted_map,
                                                             aux_mat,
                                                             num_bin_x,
                                                             num_bin_y,
                                                             num_bin_z,
                                                             mov_rhs - mov_lhs);
            // mov_density_map.mul_(util_weight);  // FIXME:
            // int macro_num = torch::sum(macro_mask).item<int>();
            // at::Tensor aux_mat3 = torch::zeros_like(init_density_map);
            // at::Tensor marco_density_map =
            //     macro_density_map_forward(normalize_node_info,
            //                         mov_conn_sorted_map,
            //                         aux_mat3,
            //                         num_bin_x,
            //                         num_bin_y,
            //                         num_bin_z,
            //                         macro_num,
            //                         mov_rhs - mov_lhs);
            overflow = overflow_fn(mov_density_map);
            // mov_density_map = mov_density_map + marco_density_map;
            at::Tensor aux_mat2 = torch::zeros_like(init_density_map);
            at::Tensor filler_density_map;
            // FIXME: is not None -> .numel()
            if (filler_sorted_map.numel()) {
                filler_density_map = density_map_forward(normalize_node_info.index({Slice(mov_rhs)}),
                                                         filler_sorted_map,
                                                         aux_mat2,
                                                         num_bin_x,
                                                         num_bin_y,
                                                         num_bin_z,
                                                         num_nodes - (mov_rhs - mov_lhs));
                // cout << filler_density_map.sizes() << endl;
                density_map = mov_density_map + filler_density_map;
            } else {
                density_map = mov_density_map;
            }
            // cout << density_map.min() << endl;
            // cout << density_map.max() << endl;
            // cout << density_map.mean() << endl;

        } else {
            overflow = node_size.new_empty(0);
            at::Tensor aux_mat = init_density_map.clone();
            density_map = density_map_forward(
                normalize_node_info, mov_sorted_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, num_nodes);
        }

        // density_map.index({"...", Slice(0, num_bin_z / 2)}) *= node_util_weight[0]; // TODO:
        // density_map.index({"...", Slice(0, num_bin_z / 2)}) *= torch::pow(node_util_weight[0], 0.5);
        // density_map -= density_map.mean(); # we don't need this one anymore

        auto [potential_scale,
              potential_coeff,
              force_x_scale,
              force_y_scale,
              force_z_scale,
              force_x_coeff,
              force_y_coeff,
              force_z_coeff] = fft_scale;
        auto [grad_mat, potential_map] = torch_dct_idct(density_map, fft_scale);

        auto energy = (potential_map * density_map).sum();
        ctx->save_for_backward({normalize_node_info, mov_sorted_map, grad_mat, density_weight_local});

        // FIXME: 4 times smaller than cuda dct
        return {energy, overflow};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
        // Use data saved in forward
        auto normalize_node_info = ctx->get_saved_variables()[0];
        auto mov_sorted_map = ctx->get_saved_variables()[1];
        auto grad_mat = ctx->get_saved_variables()[2];
        auto density_weight_local = ctx->get_saved_variables()[3];
        int num_bin_x = ctx->saved_data["num_bin_x"].toInt();
        int num_bin_y = ctx->saved_data["num_bin_y"].toInt();
        int num_bin_z = ctx->saved_data["num_bin_z"].toInt();
        int num_nodes = ctx->saved_data["num_nodes"].toInt();
        int mov_lhs = ctx->saved_data["mov_lhs"].toInt();
        int mov_rhs = ctx->saved_data["mov_rhs"].toInt();

        auto energy_grad_out = grad_outputs[0];
        auto overflow_grad_out = grad_outputs[1];
        auto energy_grad_out_xy =
            grad_outputs[0] / st::setting.cache_density_weight * st::setting.cache_density_weight_xy;
        // grad_mat: 3, 512, 512, 10
        {
            torch::NoGradGuard no_grad;
            grad_mat.slice(0, 0, 2) = grad_mat.slice(0, 0, 2) * energy_grad_out_xy;
            grad_mat.slice(0, 2, 3) = grad_mat.slice(0, 2, 3) * energy_grad_out;
        }

        // grad_mat = grad_mat * energy_grad_out;
        double grad_weight = -1.0;                                             // Gradient descent
        at::Tensor node_grad = normalize_node_info.new_zeros({num_nodes, 3});  // TODO: 3 channels

        node_grad = density_map_backward(normalize_node_info,
                                         grad_mat,
                                         mov_sorted_map,
                                         node_grad,
                                         grad_weight,
                                         num_bin_x,
                                         num_bin_y,
                                         num_bin_z,
                                         num_nodes);

        node_grad *= density_weight_local.unsqueeze(1);
        // node_grad.index({"...", 2}) *= st::setting.iteration == 0 ? 1 : 1;  // FIXME:
        // node_grad.index({Slice(mov_lhs, mov_rhs), Slice(0, 2)}) *=
        //     st::setting.iteration == 0 ? 1 : st::setting.force_coeff_2d;  // FIXME:

        // // cout << " den node grad \n" << node_grad.mean(0) << endl;
        // // cout << node_grad.mean(0) << endl;
        // cout << "========den========\n";
        // cout << node_grad[169][2] << endl;

        // Use data saved in forward
        return {node_grad,
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

class ElectronicDensityLayer : public torch::nn::Module {
public:
    ElectronicDensityLayer(at::Tensor _unit_len,
                           int _num_bin_x,
                           int _num_bin_y,
                           int _num_bin_z,
                           torch::Device device,
                           tuple<int, int, function<at::Tensor(at::Tensor)>> _overflow_helper,
                           at::Tensor _expand_ratio,
                           tuple<at::Tensor, at::Tensor, at::Tensor> _sorted_maps,
                           torch::Tensor _macro_mask,
                           bool _inference_mode = true,
                           bool _scale_w_k = true);

public:
    at::Tensor get_cache_var(at::Tensor node_pos, at::Tensor node_size, at::Tensor node_weight);
    void preprocess_inference_var(at::Tensor node_pos, at::Tensor node_size, at::Tensor node_weight);
    tuple<torch::Tensor, torch::Tensor> direct_calc_overflow(torch::Tensor node_pos,
                                                             torch::Tensor node_size,
                                                             torch::Tensor init_density_map);
    //    // FIXME: !=None -> !.numel()
    //    torch::Tensor node_weight);

public:
    variable_list forward(at::Tensor node_pos,
                          at::Tensor node_size,
                          at::Tensor init_density_map,
                          at::Tensor node_weight,
                          at::Tensor macro_mask,
                          bool calc_overflow = true) {
        auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;
        auto node_pos_channel = node_pos.index({Slice(mov_lhs, mov_rhs), 2});
        auto slicer = die_info[2 * 2 + 1] * 0.5;
        auto parter = (node_pos_channel > slicer);
        node_die = torch::_cast_Int(parter);  // FIXME:
        auto node_size_morm_util = node_size.clone();

        auto ratio = node_pos_channel / die_info[2 * 2 + 1];
        ratio = (-(1.5 + 2 * macro_mask) + ratio * (4 + macro_mask * 4)).unsqueeze(1);
        ratio = ratio.clamp(0,1);
        // mov_node_size = mov_node_size_top * ratio + mov_node_size_bot * (1-ratio);
        node_size_morm_util.slice(0, mov_lhs, mov_rhs).slice(1, 0, 2) *= (ratio_difference * (1 - ratio) + ratio);

        // FIXME:
        // node_size_morm_util.index({Slice(mov_lhs, mov_rhs), 1}) *= node_util_weight.index_select(0, node_die);
        // node_size_morm_util.index({Slice(mov_lhs, mov_rhs), 1}) *= node_util_weight_y.index_select(0, node_die);
        // node_size_morm_util.index({Slice(mov_lhs, mov_rhs), 0}) *= node_util_weight_x.index_select(0, node_die);

        node_weight = get_cache_var(node_pos, node_size_morm_util, node_weight);

        int num_nodes = node_pos.sizes()[0];
        auto grad_out = ElectronicDensityFunction::apply(node_pos,
                                                         node_size_morm_util,
                                                         node_weight,
                                                         density_weight_local,
                                                         expand_ratio,
                                                         unit_len,
                                                         init_density_map,
                                                         num_bin_x,
                                                         num_bin_y,
                                                         num_bin_z,
                                                         num_nodes,
                                                         fft_scale,
                                                         overflow_helper,
                                                         sorted_maps,
                                                         calc_overflow,
                                                         macro_mask);
        return grad_out;
    };

    torch::Tensor density_grad(at::Tensor node_pos, at::Tensor node_size, at::Tensor density_map);
    torch::Tensor density_weight_local;

public:
    /* global prams */
    int num_bin_x;
    int num_bin_y;
    int num_bin_z;
    at::Tensor macro_mask;
    bool scale_w_k;
    bool inference_mode;
    bool has_setup_inference_var = false;

    /* tensor ralated */
    at::Tensor unit_len;
    at::Tensor expand_ratio;
    tuple<double, double, function<at::Tensor(at::Tensor)>> overflow_helper;
    tuple<at::Tensor, at::Tensor, at::Tensor> sorted_maps;
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale;

    at::Tensor min_node_w;
    at::Tensor min_node_h;
    at::Tensor min_node_d;  // TODO: depth

    at::Tensor die_info;
    at::Tensor node_die;
    at::Tensor node_util_weight;
    at::Tensor node_util_weight_y;
    at::Tensor node_util_weight_x;
    at::Tensor ratio_difference;

public:
    /* cache */
    at::Tensor cache_node_weight;
};  // END MODULE

//---------------------------------------------------------------------

}  // namespace GP3D