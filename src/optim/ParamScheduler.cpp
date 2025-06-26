#include "ParamScheduler.h"

MetricRecorder::MetricRecorder(const vector<string>& names) {
    for (auto& name : names) {
        auto mi = metrics.find(name);
        if (mi == metrics.end()) {
            metrics[name] = vector<double>();
        } else {
            logger.warning("Metric %s in recorder redefined.", name.c_str());
        }
    }
}

void MetricRecorder::push(const unordered_map<string, double>& name2value) {
    for (auto& [name, value] : name2value) {
        auto mi = metrics.find(name);
        if (mi != metrics.end()) {
            mi->second.emplace_back(value);
        } else {
            logger.warning("Cannot find metric %s in recorder.", name.c_str());
        }
    }
};

double MetricRecorder::get(const string& name, int idx) {
    auto mi = metrics.find(name);
    if (mi != metrics.end()) {
        int rolledIdx = idx >= 0 ? idx : mi->second.size() + idx;
        if (rolledIdx >= 0) {
            return mi->second[rolledIdx];
        } else {
            logger.error("Incorred rolledIdx in metric %s idx %d", name.c_str(), idx);
            exit(1);
        }
    } else {
        logger.error("Cannot find metric %s in recorder.", name.c_str());
        exit(1);
    }
}

vector<double>& MetricRecorder::get_metric(const string& name) {
    auto mi = metrics.find(name);
    if (mi != metrics.end()) {
        return mi->second;
    } else {
        logger.error("Cannot find metric %s in recorder.", name);
        exit(1);
    }
}

ParamScheduler::ParamScheduler(NodeData& data) : recorder(_metrics) {
    early_stop_check_plateau = st::setting.early_stop_check_plateau;
    logger.info("Check plateau: %d", early_stop_check_plateau);
    iter = 0;
    mov_node_to_num_pins = data.mov_node_to_num_pins.clone().to(torch::kCPU);
    mov_node_area = data.mov_node_area.clone().to(torch::kCPU);
    auto [mov_lhs, mov_rhs] = data.movable_index;
    node_size = data.node_size.index({Slice(mov_lhs, mov_rhs)}).clone().to(torch::kCPU);

    best_metric = {{"overflow", double_inf}, {"hpwl", double_inf}};
    best_metric_aux = {{"overflow", double_inf}, {"hpwl", double_inf}};
    best_metric_rollback = {{"overflow", double_inf}, {"hpwl", double_inf}};

    best_sol = node_size.new_empty(0);
    best_sol_aux = node_size.new_empty(0);
    best_sol_rollback = node_size.new_empty(0);

    precond_coef = 1.0;
    precond_weight = torch::empty_like(mov_node_to_num_pins);
    density_weight = st::setting.density_weight;
    // base_gamma = st::setting.wa_coeff * torch::sum(data.unit_len).item<float>();
    base_gamma = st::setting.wa_coeff * torch::sum(data.unit_len.index({Slice(0, 2)})).item<float>();

    wa_coeff = 10 * base_gamma;
    mu = 1.0;
    max_life = 30;
    life = max_life;
    stop_overflow = st::setting.stop_overflow;
    if (st::setting.skip_2d) {
        stop_overflow = st::setting.stop_overflow_via;
    }

    use_precond = st::setting.use_precond;
    skip_update = false;
    enable_skip_update = st::setting.enable_skip_update;
    enable_fence = data.enable_fence;

    force_ratio = 0.0;
    weighted_weight = 0.0;

    // quad penalty
    // density_quad_coeff = 2000;
    density_quad_coeff = st::setting.quad_coeff;
    mov_node_weights = data.mov_node_weights.numel() ? data.mov_node_weights.clone().to(torch::kCPU).unsqueeze(2)
                                                    : mov_node_weights;

    density_weights.resize(st::setting.num_den_layer, density_weight);
    wa_coeffs.resize(st::setting.num_den_layer, wa_coeff);
    
}

void ParamScheduler::set_init_param(double init_density_weight) {
    density_weight = density_weight * init_density_weight;
    density_weights[0] = density_weight * init_density_weight;
    density_weights[1] = density_weight * init_density_weight;
    density_weights[2] = density_weight * init_density_weight;
    update_precond_weight();
}

void ParamScheduler::push_metric(double hpwl, double overflow, int iteration) {
    const unordered_map<string, double> metrics_dict = {
        {"hpwl", hpwl},
        {"overflow", overflow},
        {"mu", mu},
        {"wa_coeff", wa_coeff},
        {"density_weight", density_weight},
        {"precond_coef", precond_coef},
        {"weighted_weight", weighted_weight},
        {"force_ratio", force_ratio},
        {"iteration", iteration}
    };
    recorder.push(metrics_dict);
}

bool ParamScheduler::update_best_sol(torch::Tensor sol) {
    bool update_flag = false;
    double hpwl = recorder.get("hpwl", -1);
    double overflow = recorder.get("overflow", -1);
    int iteration = recorder.get("iteration", -1);
    // if (iter < 50) {
    //     return update_flag;
    // }

    if (overflow < stop_overflow) {
        life -= 1;
        if (life == max_life - 1) {
            // release memory of rollback solution
            best_sol_rollback = node_size.new_empty(0);
            best_metric_rollback["overflow"] = double_inf;
            best_metric_rollback["hpwl"] = double_inf;
            best_metric_rollback["iteration"] = -1;
        }
    }

    if (overflow < stop_overflow * 5 && overflow >= stop_overflow && life == max_life) {
        if (hpwl < best_metric_rollback["hpwl"] * 1.01 && overflow < best_metric_rollback["overflow"]) {
            if (best_sol_rollback.numel() == 0) {
                best_sol_rollback = sol.detach().clone();
            } else {
                best_sol_rollback.data().copy_(sol.data());
            }
            best_metric_rollback["hpwl"] = hpwl;
            best_metric_rollback["overflow"] = overflow;
            best_metric_rollback["iteration"] = iteration;
        }
        update_flag = true;
    }

    if (overflow < stop_overflow && hpwl < best_metric_aux["hpwl"] * 1.001 && overflow <= best_metric_aux["overflow"]) {
        if (best_sol_aux.numel() == 0) {
            best_sol_aux = sol.detach().clone();

        } else {
            best_sol_aux.data().copy_(sol.data());
        }
        best_metric_aux["hpwl"] = hpwl;
        best_metric_aux["overflow"] = overflow;
        best_metric_aux["iteration"] = iteration;
        update_flag = true;
    }

    if (overflow < stop_overflow && hpwl < best_metric["hpwl"]) {
        if (best_sol.numel() == 0) {
            best_sol = sol.detach().clone();
        } else {
            best_sol.data().copy_(sol.data());
        }
        best_metric["hpwl"] = hpwl;
        best_metric["overflow"] = overflow;
        best_metric["iteration"] = iteration;
        update_flag = true;
    }

    return update_flag;
}

bool ParamScheduler::need_to_early_stop() {
    if (iter < 100) return false;
    int ptr = iter - 1;
    double ptr_ovfl = recorder.get("overflow", ptr);

    if (!enable_fence && check_divergence(3, 0.01 * ptr_ovfl)) {
        // dead earlier
        life -= 6;
    }

    if (!enable_fence && check_stop_hpwl(10,5)) {
        // dead earlier
        life -= max_life;
    }

    if (ptr_ovfl < stop_overflow * 5 && ptr_ovfl >= stop_overflow) {
        if (early_stop_check_plateau) {
            if (check_plateau(recorder.get_metric("overflow"), 50, 0.05)) {
                logger.warning("Large plateau detected. Kill the optimization process.");
                life -= max_life;
            }
        }
    }

    if (life <= 0)
    {
        return true;
    }

    if (ptr_ovfl > recorder.get("overflow", ptr - 1) && recorder.get("hpwl", ptr) > best_metric["hpwl"] * 2) {
        return true;
    }

    return false;
}

bool ParamScheduler::check_plateau(std::vector<double>& x, int window, double threshold) {
    if (x.size() < window) return false;
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tensor_x = torch::from_blob(x.data(), {static_cast<long>(x.size())}, options);
    auto sliced_x = tensor_x.index({torch::indexing::Slice(tensor_x.size(0) - window, torch::indexing::None)});
    return ((torch::max(sliced_x) - torch::min(sliced_x)) / torch::mean(sliced_x) < threshold).item().toBool();
}

bool ParamScheduler::check_divergence(int window, double threshold) {
    if (best_metric["hpwl"] == double_inf) return false;
    if (iter <= window) return false;
    auto& hpwls = recorder.get_metric("hpwl");
    auto& ovfls = recorder.get_metric("overflow");
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tensor_hpwls = torch::from_blob(hpwls.data(), {static_cast<long>(hpwls.size())}, options);
    auto x = tensor_hpwls.index({torch::indexing::Slice(-window, torch::indexing::None)});

    double wl_mean = torch::mean(x).item().toDouble();
    double wl_ratio = (wl_mean - best_metric["hpwl"]) / best_metric["hpwl"];
    if (wl_ratio > threshold * 1.2) {
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        auto tensor_ovfls = torch::from_blob(ovfls.data(), {static_cast<long>(ovfls.size())}, options);
        auto y = tensor_ovfls.index({torch::indexing::Slice(-window, torch::indexing::None)});

        double overflow_mean = torch::mean(y).item().toDouble();
        auto y_lhs = y.index({torch::indexing::Slice(1, torch::indexing::None)});
        auto y_rhs = y.index({torch::indexing::Slice(torch::indexing::None, -1)});
        double overflow_diff =
            torch::sum(torch::sign(y_lhs - y_rhs) > 0).item().toDouble() / static_cast<double>(ovfls.size() - 1);
        double overflow_range = (torch::max(y) - torch::min(y)).item().toDouble();
        double overflow_ratio = (overflow_mean - max(stop_overflow, best_metric["overflow"])) / best_metric["overflow"];
        if (overflow_ratio > threshold) {
            logger.warning("Divergence detected: overflow increases too much than best overflow (%.4f > %.4f)",
                           overflow_ratio,
                           threshold);
            return true;
        } else if (overflow_range / overflow_mean < threshold) {
            logger.warning(
                "Divergence detected: overflow plateau (%.4f < %.4f)", overflow_range / overflow_mean, threshold);
            return true;
        } else if (overflow_diff > 0.6) {
            logger.warning("Divergence detected: overflow fluctuate too frequently (%.2f > 0.6)", overflow_diff);
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool ParamScheduler::check_stop_hpwl(int window, double threshold) {
    double min_hpwl = best_metric_aux["hpwl"];
    if(best_metric_rollback["hpwl"]<min_hpwl)
    {
        min_hpwl = best_metric_rollback["hpwl"];
    }
    if(best_metric["hpwl"]<min_hpwl)
    {
        min_hpwl = best_metric["hpwl"];
    }
    if (min_hpwl == double_inf) return false;
    if (iter <= window) return false;
    auto& hpwls = recorder.get_metric("hpwl");
    auto& ovfls = recorder.get_metric("overflow");
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tensor_hpwls = torch::from_blob(hpwls.data(), {static_cast<long>(hpwls.size())}, options);
    auto x = tensor_hpwls.index({torch::indexing::Slice(-window, torch::indexing::None)});

    double wl_mean = torch::mean(x).item().toDouble();
    double wl_ratio = (wl_mean - min_hpwl) / min_hpwl;
    if (wl_ratio > threshold) {
        logger.warning("Stop because hpwl is too high");
        return true;
    } else {
        return false;
    }
}

tuple<torch::Tensor, double, double, int> ParamScheduler::get_best_solution() {
    torch::Tensor my_best_sol = node_size.new_empty(0);
    double my_best_hpwl = double_inf;
    double my_best_overflow = double_inf;
    int solution_type = 0;
    int my_best_iteration = -1;
    if (best_sol_rollback.numel() != 0) {
        // have solution
        my_best_sol = best_sol_rollback.data();
        my_best_hpwl = best_metric_rollback["hpwl"];
        my_best_overflow = best_metric_rollback["overflow"];
        my_best_iteration = best_metric_rollback["iteration"];
        solution_type = 3;
    } else if (best_sol.numel() == 0 && best_sol_aux.numel() == 0) {
        solution_type = 0;
    } else if (best_sol_aux.numel() == 0) {
        my_best_sol = best_sol.data();
        my_best_hpwl = best_metric["hpwl"];
        my_best_overflow = best_metric["overflow"];
        my_best_iteration = best_metric["iteration"];
        solution_type = 1;
    } else if (best_sol.numel() == 0) {
        my_best_sol = best_sol_aux.data();
        my_best_hpwl = best_metric_aux["hpwl"];
        my_best_overflow = best_metric_aux["overflow"];
        my_best_iteration = best_metric_aux["iteration"];
        solution_type = 2;
    } else {
        if (best_metric_aux["hpwl"] < best_metric["hpwl"] * 1.001 &&
            best_metric_aux["overflow"] * 1.1 < best_metric["overflow"]) {
            my_best_sol = best_sol_aux.data();
            my_best_hpwl = best_metric_aux["hpwl"];
            my_best_overflow = best_metric_aux["overflow"];
            my_best_iteration = best_metric_aux["iteration"];
            solution_type = 2;
        } else {
            my_best_sol = best_sol.data();
            my_best_hpwl = best_metric["hpwl"];
            my_best_overflow = best_metric["overflow"];
            my_best_iteration = best_metric["iteration"];
            solution_type = 1;
        }
    }

    if (solution_type == 0) {
        logger.info("Cannot find best solution. Use the last solution.");
    } else if (solution_type == 1) {
        logger.info("Find best solution (type %d HPWL driven) masked_hpwl: %.4E overflow: %.4f, iteration: %d",
                    solution_type,
                    my_best_hpwl,
                    my_best_overflow,
                    my_best_iteration);
    } else if (solution_type == 2) {
        logger.info("Find best solution (type %d OVFL driven) masked_hpwl: %.4E overflow: %.4f, iteration: %d",
                    solution_type,
                    my_best_hpwl,
                    my_best_overflow,
                    my_best_iteration);
    } else if (solution_type == 3) {
        logger.info("Cannot find best solution. Use roll back solution (type %d) masked_hpwl: %.4E overflow: %.4f, iteration: %d",
                    solution_type,
                    my_best_hpwl,
                    my_best_overflow,
                    my_best_iteration);
    } else {
        throw std::runtime_error("Unknown solution type");
    }

    return {my_best_sol, my_best_hpwl, my_best_overflow, my_best_iteration};
}


void ParamScheduler::visualize() {
    // plt::backend("Agg");
    for (auto key : _metrics) {
        int view = iter;
        
        std::vector<float> x(view), y(view);

        for (int i = 0; i < view; ++i) {
            x.at(i) = i;
            y.at(i) = recorder.get(key, i);
        }

        // plt::plot(x, y);
        // plt::title(key);
        // vector<int> xtick = {0, (int)(view - 1)};
        // plt::xticks(xtick);

        // std::filesystem::path current_dir(std::filesystem::current_path());
        // std::filesystem::path result_dir(st::setting.result_dir);
        // std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path eval(std::string("eval"));
        std::filesystem::path fig_root = logger.res_root / eval;
        if (!std::filesystem::exists(fig_root)) {
            std::filesystem::create_directories(fig_root);
        }
        // cout << res_root << endl;
        std::string fig_name = string("/ms_2.5_" + key + ".png");
        cout << fig_name << endl;

        // plt::save(fig_path);
        // plt::close();

        plot_pt(y,
                (char *)"imshow",
                (char *)fig_root.string().c_str(),
                (char *)fig_name.c_str());
    }
}
