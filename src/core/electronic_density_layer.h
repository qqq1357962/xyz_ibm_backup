#pragma once

#include "dct.h"
#include "density_map/density_map.h"
#include "density_smooth/density_smooth.h"
#include "global.h"
#include "../placer/visualization.h"

using namespace torch::autograd;

class ElectronicDensityFunction : public Function<ElectronicDensityFunction> {
public:
    static variable_list forward(
        AutogradContext *ctx,
        at::Tensor node_pos,
        at::Tensor node_size,
        at::Tensor node_weight,
        at::Tensor expand_ratio,
        at::Tensor unit_len,
        at::Tensor init_density_map,
        int num_bin_x,
        int num_bin_y,
        int num_nodes,
        tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale,
        tuple<double, double, std::function<at::Tensor(at::Tensor)>> overflow_helper,
        tuple<at::Tensor, at::Tensor, at::Tensor> sorted_maps,
        bool calc_overflow,
        at::Tensor macro_mask) {
        // Save data for backward in context
        ctx->saved_data["num_bin_x"] = num_bin_x;
        ctx->saved_data["num_bin_y"] = num_bin_y;
        ctx->saved_data["num_nodes"] = num_nodes;

        auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;
        auto [mov_sorted_map, mov_conn_sorted_map, filler_sorted_map] = sorted_maps;

        at::Tensor density_map;
        at::Tensor overflow;
        at::Tensor normalize_node_info;
        // 1) Compute Density Map
        normalize_node_info = node_size.new_empty({num_nodes, 5});  // x_l, x_h, y_l, y_h, weight
        normalize_node_info = density_map_normalize_node(node_pos,
                                                         node_size,
                                                         node_weight,
                                                         expand_ratio,
                                                         unit_len,
                                                         normalize_node_info,
                                                         num_bin_x,
                                                         num_bin_y,
                                                         num_nodes);
        if (calc_overflow) {
            at::Tensor aux_mat = init_density_map.clone();
            at::Tensor mov_density_map = density_map_forward(normalize_node_info.index({Slice(mov_lhs, mov_rhs)}),
                                                             mov_conn_sorted_map,
                                                             aux_mat,
                                                             num_bin_x,
                                                             num_bin_y,
                                                             mov_rhs - mov_lhs);
            
            // at::Tensor marco_indices = torch::nonzero(macro_mask).squeeze();
            // int macro_num = torch::sum(macro_mask).item<int>();
            // // cout << "total macros" << macro_num << endl;
            // at::Tensor aux_mat3 = torch::zeros_like(init_density_map);
            // at::Tensor marco_density_map =
            //     macro_density_map_forward(normalize_node_info,
            //                         mov_conn_sorted_map.index({Slice(mov_lhs, macro_num)}),
            //                         aux_mat3,
            //                         num_bin_x,
            //                         num_bin_y,
            //                         macro_num);
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
                                                         num_nodes - (mov_rhs - mov_lhs));
                density_map = mov_density_map + filler_density_map;
            } else {
                density_map = mov_density_map;
            }

        } else {
            overflow = node_size.new_empty(0);
            at::Tensor aux_mat = init_density_map.clone();
            density_map =
                density_map_forward(normalize_node_info, mov_sorted_map, aux_mat, num_bin_x, num_bin_y, num_nodes);
        }

        // propogate to backward layer
        auto [potential_scale, potential_coeff, force_x_scale, force_y_scale, force_x_coeff, force_y_coeff] = fft_scale;
        auto [grad_mat, potential_map] = torch_dct_idct(density_map, fft_scale);

#ifdef DEBUG
        //if ((st::setting.iteration % st::setting.log_freq) == 0) {
        if ((false) {
            cout << density_map0.sum().item<float>() << " | " << density_map1.sum().item<float>() << endl;
            cout << overflow_fn(density_map0).item<float>() << " | " << overflow_fn(density_map1).item<float>() << endl;

            std::filesystem::path fig_path = logger.res_root;
            plot_pt(density_map0),
                    (char *)"plot",
                    (char *)fig_path.string().c_str(),
                    (char *)("test_" + std::to_string(st::setting.iteration) + ".png").c_str());
            plot_pt(density_map1),
                    (char *)"plot",
                    (char *)fig_path.string().c_str(),
                    (char *)("test_smooth_" + std::to_string(st::setting.iteration) + ".png").c_str());
        }
#endif  // DEBUG

        auto energy = (potential_map * density_map).sum();
        ctx->save_for_backward({normalize_node_info, mov_sorted_map, grad_mat});

        // FIXME: 4 times smaller than cuda dct
        return {energy, overflow};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
        // Use data saved in forward
        auto normalize_node_info = ctx->get_saved_variables()[0];
        auto mov_sorted_map = ctx->get_saved_variables()[1];
        auto grad_mat = ctx->get_saved_variables()[2];
        int num_bin_x = ctx->saved_data["num_bin_x"].toInt();
        int num_bin_y = ctx->saved_data["num_bin_y"].toInt();
        int num_nodes = ctx->saved_data["num_nodes"].toInt();

        auto energy_grad_out = grad_outputs[0];
        auto overflow_grad_out = grad_outputs[1];
        
        //@@ grad_mat: (2,512,1024)??? 2, num_bin_x, num_bin_y
        int size1 = grad_mat.size(1);
        int size2 = grad_mat.size(2);
        torch::Tensor zeros_tensor = torch::zeros({1, size1, size2}).to(grad_mat.device());
        
        torch::Tensor new_tensor = torch::cat({grad_mat, zeros_tensor}, 0);
        // saveChannels(grad_mat, "grad_mat");
        string info = "grad_mat";
        // saveChannels(new_tensor, "grad_mat");
        // plot_pt(new_tensor, "save", "./", "0")
        grad_mat = grad_mat * energy_grad_out;
        double grad_weight = -1.0;  // Gradient descent
        at::Tensor node_grad = normalize_node_info.new_zeros({num_nodes, 2});
        
        // std::cout << "node_grad mean flag0" << node_grad.mean(0) << endl;

        at::Tensor node_grad_4part = normalize_node_info.new_zeros({num_nodes, 4});
        at::Tensor macro_mask = st::setting.cache_macro_mask;
        // std::cout << "before backward macro_mask.sizes()"<< macro_mask.sizes() << " node_grad.sizes() " << node_grad.sizes() << std::endl;
        node_grad = density_map_backward(
            normalize_node_info, grad_mat, mov_sorted_map, node_grad, node_grad_4part, macro_mask, grad_weight, num_bin_x, num_bin_y, num_nodes);
        
        // std::cout << "end backward1 "<< std::endl;

        // if (node_grad.defined()) {
        //     std::cout << "node_grad is defined" << std::endl;
        //     std::cout << "node_grad size: " << node_grad.sizes() << std::endl;
        //     std::cout << "node_grad device: " << node_grad.device() << std::endl;
        //     try {
        //         std::cout << "node_grad mean: " << node_grad.mean(0) << std::endl;
        //     } catch (const std::exception& e) {
        //         std::cout << "Error calculating mean: " << e.what() << std::endl;
        //     }
        // } else {
        //     std::cout << "node_grad is undefined!" << std::endl;
        // }

        // st::setting.cache_density_grad_4part = node_grad_4part.cpu();//@@@
        // std::cout << "node_grad mean" << node_grad.mean(0) << endl;
        

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
                Variable(),
                Variable(),};
    }
};  // END MODULE

//---------------------------------------------------------------------

class ElectronicDensityLayer : public torch::nn::Module {
public:
    ElectronicDensityLayer(at::Tensor _unit_len,
                           int _num_bin_x,
                           int _num_bin_y,
                           torch::Device device,
                           tuple<double, double, function<at::Tensor(at::Tensor)>> _overflow_helper,
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
    torch::Tensor density_grad(at::Tensor node_pos,
                               at::Tensor node_size,
                               at::Tensor node_weight,
                               at::Tensor density_map);

public:
    variable_list forward(at::Tensor node_pos,
                          at::Tensor node_size,
                          at::Tensor init_density_map,
                          // FIXME: !=None -> !.numel()
                          at::Tensor node_weight,
                          bool calc_overflow = true) {
        node_weight = get_cache_var(node_pos, node_size, node_weight);
        int num_nodes = node_pos.sizes()[0];
        auto grad_out = ElectronicDensityFunction::apply(node_pos,
                                                         node_size,
                                                         node_weight,
                                                         expand_ratio,
                                                         unit_len,
                                                         init_density_map,
                                                         num_bin_x,
                                                         num_bin_y,
                                                         num_nodes,
                                                         fft_scale,
                                                         overflow_helper,
                                                         sorted_maps,
                                                         calc_overflow,
                                                         macro_mask);
        return grad_out;
    };

public:
    /* global prams */
    int num_bin_x;
    int num_bin_y;
    at::Tensor macro_mask;
    bool scale_w_k;
    bool inference_mode;
    bool has_setup_inference_var = false;

    /* tensor ralated */
    at::Tensor unit_len;
    at::Tensor expand_ratio;
    tuple<double, double, function<at::Tensor(at::Tensor)>> overflow_helper;
    tuple<at::Tensor, at::Tensor, at::Tensor> sorted_maps;
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale;

    at::Tensor min_node_w;
    at::Tensor min_node_h;

public:
    /* cache */
    at::Tensor cache_node_weight;
};  // END MODULE

//---------------------------------------------------------------------