#include "ParamScheduler.h"



void ParamScheduler::step(double hpwl, double overflow, torch::Tensor node_pos) {
    update_precond_weight();
    push_metric(hpwl, overflow, iter);
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
    step_density_weight();
    step_wa_coeff();
    step_precond_coef();
    iter += 1;
}

void ParamScheduler::step_density_weight() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    double delta_hpwl = recorder.get("hpwl", -1) - recorder.get("hpwl", -2);
    if (delta_hpwl < 0) {
        mu = st::setting.density_weight_coef * std::max(std::pow(0.9999, static_cast<double>(iter)), 0.98);
    } else {
        mu = st::setting.density_weight_coef * std::clamp(std::pow(1.05, -delta_hpwl / st::setting.magic_hpwl), 0.95, 1.05);
    }
    density_weight *= mu;
}

void ParamScheduler::step_wa_coeff() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    double coef = std::pow(10, (recorder.get("overflow", -1) - 0.1) * 20 / 9 - 1);
    wa_coeff = coef * base_gamma;
}

void ParamScheduler::step_precond_coef() {
    if (!use_precond) return;
    // logger.info("precond_coef: %lf", precond_coef);//@@
    if (recorder.get("overflow", iter) < 0.15 && precond_coef < 1024) {
        if (iter % 20 == 0) {
            precond_coef *= 2;
        }
    }
}

void ParamScheduler::update_precond_weight() {
    if (!use_precond) return;

    torch::Tensor alpha_1 = mov_node_to_num_pins;
    torch::Tensor alpha_2 = precond_coef * density_weight * mov_node_area;
    precond_weight = (alpha_1 + alpha_2).clamp_(1.0);
    /////////
    // torch::Tensor large_node_mask = (mov_node_area > 100000);
    // torch::Tensor precond_weight_temp = alpha_1 + alpha_2;
    // precond_weight = torch::where(large_node_mask, precond_weight_temp.clamp(1.0), precond_weight_temp);
    /////////
    torch::Tensor a2_norm = alpha_2.norm(1);
    weighted_weight = (a2_norm / (alpha_1.norm(1) + a2_norm)).item().toFloat();
}



// TODO: multi ps ===============================================================

void ParamScheduler::steps(torch::Tensor hpwls, torch::Tensor overflows, torch::Tensor node_pos) {
    double hpwl = hpwls.sum().item<double>();
    double overflow = overflows.index({Slice(0, 2)}).mean().item<float>();
    // double overflow = overflows[0].item<float>() * 0.45 + overflows[1].item<float>() * 0.45 + overflows[2].item<float>() * 0.1;
    // double overflow = overflows.index({Slice(0, 1)}).mean().item<float>();
    push_metric(hpwl, overflow, iter);
    cur_hpwls = hpwls;
    cur_overflows = overflows;

    update_precond_weights();
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
    step_density_weights();
    step_wa_coeffs();
    step_precond_coef();
    prev_hpwls = hpwls;
    prev_overflows = overflows;
    iter += 1;
}

void ParamScheduler::step_density_weights() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;

    // mu = 1;
    // double mu1;
    // for (int i = 0; i < 2; i++) {
    //     double delta_hpwl = (cur_hpwls[i] - prev_hpwls[i]).item<double>();
    //     if (delta_hpwl < 0) {
    //     mu1 = st::setting.density_weight_coef * std::max(std::pow(0.9999, static_cast<double>(iter)), 0.98);
    //     } else {
    //         mu1 = st::setting.density_weight_coef * std::clamp(std::pow(1.05, -delta_hpwl / st::setting.magic_hpwl), 0.95, 1.05);
    //     }
    //     density_weights[i] *= mu1;
    //     mu *= std::sqrt(mu1);
    //     density_weights[2] *= std::sqrt(mu1);
    // }
    // density_weight = 1;

    double delta_hpwl = (cur_hpwls - prev_hpwls).sum().item<double>();
    if (delta_hpwl < 0) {
        mu = st::setting.density_weight_coef * std::max(std::pow(0.9999, static_cast<double>(iter)), 0.98);
    } else {
        mu = st::setting.density_weight_coef * std::clamp(std::pow(1.05, -delta_hpwl / st::setting.magic_hpwl), 0.95, 1.05);
    }
    density_weight *= mu;
    density_weights[0] *= mu;
    density_weights[1] *= mu;
    density_weights[2] *= mu;
}

void ParamScheduler::step_wa_coeffs() {
    if (iter < 1) return;
    if (enable_skip_update && skip_update) return;
    // for (int i = 0; i < 2; i++) {
    //     double coef = std::pow(10, (cur_overflows[i].item<double>() - 0.1) * 20 / 9 - 1);
    //     wa_coeffs[i] = coef * base_gamma;
    // }
    // wa_coeff = (wa_coeffs[0] + wa_coeffs[1]) / 2;
    double coef = std::pow(10, (cur_overflows.index({Slice(0, 2)}).mean()
                                .item<double>() - 0.1) * 20 / 9 - 1);
    wa_coeff = coef * base_gamma;
    wa_coeffs[0] = coef * base_gamma;
    wa_coeffs[1] = coef * base_gamma;
}

void ParamScheduler::update_precond_weights() {
    if (!use_precond) return;

    torch::Tensor alpha_1 = mov_node_to_num_pins;
    torch::Tensor alpha_2 = torch::zeros_like(mov_node_to_num_pins);

    for (int i = 0; i < st::setting.num_den_layer; i++)
        alpha_2 += precond_coef * density_weights[i] * mov_node_area * mov_node_weights[i];

    precond_weight = (alpha_1 + alpha_2).clamp_(1.0);
    torch::Tensor a2_norm = alpha_2.norm(1);
    weighted_weight = (a2_norm / (alpha_1.norm(1) + a2_norm)).item().toFloat();
}
