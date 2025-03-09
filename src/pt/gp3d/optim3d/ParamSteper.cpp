 #include "ParamScheduler.h"



namespace GP3D {

void ParamScheduler::step(double hpwl, double hpwl_xy, double overflow, torch::Tensor node_pos) {
    // if (local_density_lock) {
    //     push_metric(hpwl, overflow);
    //     update_best_sol(node_pos);
    //     iter += 1;
    //     return;
    // }
    update_precond_weight();
    push_metric(hpwl, hpwl_xy, overflow, iter);//@@@
    update_best_sol(node_pos);
    if (enable_skip_update) {
        if (weighted_weight > 0.5 && weighted_weight < 0.99) {
            skip_update = (iter % 3 != 0);
        } else if (iter < 50) {
            // slow down the param update of early stage
            skip_update = (iter % 3 != 0);
        } else {
            skip_update = false;
        }
    }
    if (!local_density_lock)
    {
        step_density_weight();
        step_density_weight_xy();
    }
    step_wa_coeff();
    if (!local_density_lock) step_precond_coef();
    iter += 1;
}

void ParamScheduler::step_density_weight() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    double delta_hpwl = recorder.get("hpwl", -1) - recorder.get("hpwl", -2);
    // double delta_overflow = recorder.get("overflow", -1) - recorder.get("overflow", -2);
    if (delta_hpwl < 0) {
        mu = 1.05 * std::max(std::pow(0.9999, static_cast<double>(iter)), 0.98);
    } else {
        mu = 1.05 * std::clamp(std::pow(1.05, -delta_hpwl / st::setting.magic_hpwl), 0.95, 1.05);
    }
    density_weight *= mu;

    // if (delta_overflow < 0) {
    //     mu = std::max(std::pow(0.99, static_cast<double>(-delta_overflow * 10 / recorder.get("overflow", -1))), 0.99);
    // }
    // density_weight *= mu;
    st::setting.cache_density_weight = density_weight;
    // net_weight_coef /= mu;
}

void ParamScheduler::step_density_weight_xy() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    double delta_hpwl_xy = recorder.get("hpwl_xy", -1) - recorder.get("hpwl_xy", -2);
    if (delta_hpwl_xy < 0) {
        mu = 1.05 * std::max(std::pow(0.9999, static_cast<double>(iter)), 0.98);
    } else {
        mu = 1.05 * std::clamp(std::pow(1.05, -delta_hpwl_xy / st::setting.magic_hpwl), 0.95, 1.05);
    }
    density_weight_xy *= mu;
    st::setting.cache_density_weight_xy = density_weight_xy;
    //@@ cout<<"density_weight_xy: "<<density_weight_xy<<endl;
    // net_weight_coef /= mu;
}

void ParamScheduler::step_wa_coeff() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    double coef = std::pow(10, (recorder.get("overflow", -1) - 0.1) * 20 / 9 - base_overflow);
    // if (recorder.get("overflow", -1) >= 0.3)
    //     coef = std::pow(10, (recorder.get("overflow", -1) - 0.1) * 20 / 9 - base_overflow);
    // else
    //     coef = std::pow(10, (recorder.get("overflow", -1) - 0.1) / 2 - base_overflow);
    // else
    //     coef = std::pow(10, (recorder.get("overflow", -1) - 0.1) / 4 - base_overflow);
    wa_coeff = coef * base_gamma;
}

// void ParamScheduler::step_wa_coeff_xy() {
//     if (iter < 1) return;
//     if (enable_skip_update && skip_update) return;
//     double coef = std::pow(10, (recorder.get("overflow_xy", -1) - 0.1) * 20 / 9 - base_overflow_xy);
//     wa_coeff_xy = coef * base_gamma_xy;
// }

void ParamScheduler::step_precond_coef() {
    if (!use_precond) return;
    if (recorder.get("overflow", iter) < 0.10 && precond_coef < 1024) {
        if (iter % 25 == 0) {
            precond_coef *= 2;
        }
    }
}

void ParamScheduler::update_precond_weight() {
    if (!use_precond) return;

    torch::Tensor alpha_1 = mov_node_to_num_pins;
    torch::Tensor alpha_2 = precond_coef * density_weight * mov_node_area;
    precond_weight = (alpha_1 + alpha_2).clamp_(1.0);
    // torch::Tensor large_node_mask = (mov_node_area > 100000);
    // torch::Tensor precond_weight_temp = alpha_1 + alpha_2;
    // precond_weight = torch::where(large_node_mask, precond_weight_temp.clamp(10.0), precond_weight_temp);
    torch::Tensor a2_norm = alpha_2.norm(1);
    weighted_weight = (a2_norm / (alpha_1.norm(1) + a2_norm)).item().toFloat();
}

void ParamScheduler::step_local_density_weight(torch::Tensor node_density) {
    // cout << node_density.mean() << endl;
    // cout << node_density.max() << endl;
    // cout << node_density.min() << endl;

    // node_density = node_density.clamp(1.0);

    density_weight_local = (density_weight_local * torch::pow(1.05, node_density - 1).clamp(1.0, 1.05));

    net_weight_coef = density_weight_local.mean(0).item<double>();

    // cout << "------------------ node density==============\n";
    // cout << node_density[169] << endl;


    // int num_bin_z = density_maps.size(2);
    // auto density_map = density_maps.index({"...", Slice(0, num_bin_z / 2)}).mean(2);
    // auto density_map1 = density_maps.index({"...", Slice(num_bin_z / 2, None)}).mean(2);
    // std::filesystem::path eval(std::string("eval"));
    // std::filesystem::path fig_root = logger.res_root / eval;
    // if (!std::filesystem::exists(fig_root)) {
    //     std::filesystem::create_directories(fig_root);
    // }

    // auto dj = torch::rot90(density_map.to(torch::kCPU));
    // plot_pt(dj,
    //         (char *)"imshow",
    //         (char *)fig_root.string().c_str(),
    //         (char *)("density_" + std::to_string(31) + ".png").c_str());

    // auto dj1 = torch::rot90(density_map1.to(torch::kCPU));
    // plot_pt(dj1,
    //         (char *)"imshow",
    //         (char *)fig_root.string().c_str(),
    //         (char *)("density_" + std::to_string(32) + ".png").c_str());


    
    // exit(1);
}

}
