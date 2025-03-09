#include "../run_placement.h"

void run_placement_main_nesterov() {
    /* General settings */
    torch::set_num_threads(st::setting.num_threads);
    setenv("OMP_NUM_THREADS", to_string(st::setting.num_threads).c_str(), true);

    /* set device */
    torch::Device device = torch::kCPU;
    logger.info("CUDA DEVICE COUNT: %d", torch::cuda::device_count());
    if (torch::cuda::is_available()) {
        logger.info("CUDA is available! Training on GPU.");
        // device = torch::kCUDA;
        device = torch::Device(torch::kCUDA, st::setting.gpu);
    }

    logger.info("Use Nesterov optimizer!");
    if (st::setting.scale_design) {
        logger.warning("Eplace's nesterov optimizer cannot support normalized die. Disable scale_design.");
        st::setting.scale_design = false;
    }

    states hpwl_state;

    /* database */
    auto [design_info, rawdb, gpdb] = load_dataset();
    NodeData data(design_info, device);
    data.preprocess();

    data.to(torch::kCPU);

    /* initialization */
    torch::Tensor init_density_map = get_init_density_map(data);
    data.init_filler();

    auto [mov_lhs, mov_rhs] = data.movable_index;
    auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info();
    mov_node_pos = mov_node_pos.to(data.device).requires_grad_(true);
    mov_node_size = mov_node_size.to(data.device);
    expand_ratio = expand_ratio.to(data.device);
    data.to(data.device);

    torch::Tensor node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4;
    torch::Tensor node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4;

    std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn = [&node_pos_lb, &node_pos_ub](torch::Tensor x) {
        x.data().clamp_(node_pos_lb, node_pos_ub);
        return x;
    };

    /* overflow function */
    std::function<torch::Tensor(torch::Tensor)> overflow_fn = [&data](torch::Tensor mov_density_map) {
        torch::Tensor overflow_sum = ((mov_density_map - st::setting.target_density) * data.bin_area).clamp_(0.0).sum();
        return overflow_sum / data.__total_mov_area_without_filler__;
    };
    auto overflow_helper = make_tuple(mov_lhs, mov_rhs, overflow_fn);
    auto density_map_layer = ElectronicDensityLayer(data.unit_len, data.num_bin_x, data.num_bin_y, device,
                                                    overflow_helper, expand_ratio, data.sorted_maps, data.macro_mask);

    if (st::setting.gp_model == "null") { /* trunc nodes */
        /* parameteer scheduler */
        ParamScheduler ps = ParamScheduler(data);

        /* objective function */
        torch::Tensor conn_fix_node_pos = data.node_pos.new_empty({0, 2});
        if (get<0>(data.fixed_connected_index) < get<1>(data.fixed_connected_index)) {
            auto [lhs, rhs] = data.fixed_connected_index;
            conn_fix_node_pos = data.node_pos.index({Slice(lhs, rhs), "..."});
        }
        conn_fix_node_pos = conn_fix_node_pos.detach();
        std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> obj_and_grad_fn =
            [&trunc_node_pos_fn, &mov_node_size, &init_density_map, &density_map_layer, &conn_fix_node_pos, &ps,
             &data](at::Tensor mov_node_pos) {
                return calc_obj_and_grad(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_map,
                                         density_map_layer, conn_fix_node_pos, ps, data);
            };

        /* evaluation function */
        std::function<tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> evaluator_fn =
            [&trunc_node_pos_fn, &mov_node_size, &init_density_map, &density_map_layer, &conn_fix_node_pos, &ps,
             &data](at::Tensor mov_node_pos) {
                return fast_evaluator(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_map,
                                      density_map_layer, conn_fix_node_pos, ps, data);
            };

        /* Nesterov optimizer */
        auto optimizer = torch::optim::Nesterov({mov_node_pos}, torch::optim::NesterovOptions(0.0), obj_and_grad_fn);
        init_params(mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, density_map_layer,
                    mov_node_size, init_density_map, optimizer, ps, data);

        double init_lr =
            estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, st::setting.lr);
        for (auto& group : optimizer.param_groups()) {
            auto& options = static_cast<torch::optim::NesterovOptions&>(group.options());
            options.set_lr(init_lr);
        }

        logger.info("start gp");
        int& iteration = st::setting.iteration;
        iteration = 0;
        for (iteration = 0; iteration < st::setting.inner_iter; iteration++) {
            torch::Tensor obj = optimizer.step();
            auto [hpwl, overflow] = evaluator_fn(mov_node_pos);
            ps.step(hpwl.item().toFloat(), overflow.item().toFloat(), mov_node_pos);
            if (ps.need_to_early_stop()) {
                break;
            }
            if (iteration % st::setting.log_freq == 0 || iteration == st::setting.inner_iter - 1) {
                logger.info(
                    "iter: %d | masked_hpwl: %.2E overflow: %.4f obj: %.4E "
                    "density_weight: %.4E wa_coeff: %.4E",
                    iteration, hpwl.item().toFloat(), overflow.item().toFloat(), obj.item().toFloat(),
                    ps.density_weight, ps.wa_coeff);

                if (st::setting.draw_placement) {
                    auto node_pos_to_draw = mov_node_pos.clone();
                    // node_pos_to_draw = torch::cat({node_pos_to_draw, data.node_pos.index({Slice(mov_rhs, torch::indexing::None)})}, 0);
                    auto node_size_to_draw = mov_node_size.clone();
                    // node_size_to_draw = torch::cat({node_size_to_draw, data.node_size.index({Slice(mov_rhs, torch::indexing::None)})}, 0);
                    auto info = make_tuple(iteration, 0, data.design_name + "_GP");
                    draw_fig_with_cairo_cpp(node_pos_to_draw.to(torch::kCPU), node_size_to_draw.to(torch::kCPU), data, info);
                }
            }
        }

        /* retrieve best score and evaluate without filler */
        auto [best_sol, best_hpwl, best_overflow, best_iteration] = ps.get_best_solution();
        if (best_sol.numel() != 0) {
            mov_node_pos.data().copy_(best_sol);
        }
        // if (st::setting.visualize_curve) ps.visualize();
        logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f", iteration, best_hpwl, best_overflow);

        auto [hpwl, overflow] =
            evaluate_placement(mov_node_pos.to(device), density_map_layer, init_density_map, data);
        logger.info("After GP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f", (hpwl).item().toFloat(),
                    overflow.item().toFloat());
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = (hpwl).item<float>();
        hpwl_state.hpwls_gp = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];

        if (true) {
            auto info = make_tuple(iteration, 0, data.design_name + "_GP");
            draw_fig_with_cairo_cpp(mov_node_pos.to(torch::kCPU), mov_node_size.to(torch::kCPU), data, info);
        }
    } else {
        logger.info("Loading model from %s", st::setting.gp_model.c_str());
        torch::load(mov_node_pos, st::setting.gp_model);
    }

    data.to(torch::kCPU);
    auto node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
    // node_pos = torch::cat({node_pos, data.node_pos.index({Slice(mov_rhs, torch::indexing::None)})}, 0);
    auto node_size = data.node_size.index({Slice(mov_lhs, mov_rhs)});
    node_size = torch::cat({node_size, data.node_size.index({Slice(mov_rhs, torch::indexing::None)})}, 0);
    data.node_die = torch::ones({data.num_nodes}, dtype(torch::kInt));

    auto [fix_lhs, fix_rhs] = data.fixed_connected_index;

    /* scale design back to original size */
    node_pos *= data.site_width;
    node_size *= data.site_width;
    data.postscale_by_site_width();

    /* save to .pt model */
    if (st::setting.save_model) {
        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string model_dir = res_root.string() + "/gp_pt.pt";
        logger.info("Save gp2d to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
    }


    // /* align to row */
    // node_pos.index({"...", 1}).data().copy_((torch::floor(node_pos.index({"...", 1}) / data.rowHeights) * data.rowHeights));

    if (true) {
        auto info = make_tuple(0, 0, data.design_name + "_GP");
        draw_fig_with_cairo_cpp(node_pos.to(torch::kCPU), node_size.to(torch::kCPU), data, info);
    }

    torch::Tensor node_weight = torch::ones(node_pos.size(0), torch::dtype(torch::kInt));
    node_weight.index_put_({Slice(mov_rhs, None)}, 0);
    if (st::setting.lg) {
        // /* Greedy Legalization */
        // greedyLegalization(node_pos.contiguous(),
        //                    node_size_legal,
        //                    node_pos_legal,
        //                    node_weight,
        //                    data.die_info,
        //                    data.numRows,
        //                    data.rowHeights,
        //                    1,
        //                    64,
        //                    node_pos.size(0));
        // if (true) {
        //     auto info = make_tuple(-1, 0, data.design_name + "_Greedy_LG");
        //     draw_fig_with_cairo_cpp(node_pos_legal, node_size_legal, data, info);
        // }

        // /* Abacus Legalization */
        // abacusLegalization(node_pos.contiguous(),
        //                    node_size_legal,
        //                    node_pos_legal,
        //                    node_weight,
        //                    data.die_info,
        //                    data.numRows,
        //                    data.rowHeights,
        //                    1,
        //                    64,
        //                    node_pos.size(0));

        logger.info("============= Cell LG =============");
        /* update node_weight for each chip */
        /* Legalization */
        torch::Tensor node_pos_lg = node_pos.clone();
        torch::Tensor node_size_lg = node_size.clone();

        dp::DetailedPlaceDataTensor lg_db_at(data, node_pos_lg, node_size_lg);

        lg_db_at.update_node_weight(node_weight);
        dp::legalizationV2(data, lg_db_at, node_pos_lg, data.numRows.item<int>(), data.rowHeights.item<float>(), 0, 1, 64,0,0);

        if (true) {
            auto info = make_tuple(-1, 0, data.design_name + "_LG_");
            draw_fig_with_cairo_cpp(node_pos_lg, node_size_lg, data, info);
        }

        node_pos = node_pos_lg;

        // auto [hpwl, overflow] =
        //     evaluate_placement(node_pos_lg.to(device), density_map_layer, init_density_map, data);
        // logger.info("After AbacusLG, exact HPWL: %.4E exact Overflow: %.4f", hpwl.item<float>(),
        //             overflow.item<float>());
        torch::Tensor conn_node_pos =
        torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})}, 0);
        torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
            conn_node_pos.to(data.device), data.pin_id2node_id.to(data.device), data.pin_rel_cpos.to(data.device));
        torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach()));
        logger.info("After LG, eval, exact HPWL: %.4E", hpwl.item().toFloat());
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl.item<float>() / (float)data.site_width;
        hpwl_state.hpwls_lg = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];
        hpwl_state.log();
    }

    /* save to .pt model */
    if (st::setting.save_model) {
        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string model_dir = res_root.string() + "/model.pt";
        logger.info("Save model to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
    }

    if (st::setting.dp) {
        /* Detailed placement */
        torch::Tensor node_pos_dp = node_pos.clone();
        torch::Tensor node_size_dp = node_size.clone();

        dp::DetailedPlaceDataTensor dp_db_at(data, node_pos_dp, node_size_dp);

        /* update node_weight for each chip */
        dp_db_at.update_node_weight(node_weight);

        dp::detail_placement(data, dp_db_at, node_pos_dp, data.numRows.item<int>(), data.rowHeights.item<float>(),
                             data.num_bin_x, data.num_bin_y);

        // auto [hpwl, overflow] =
        //     evaluate_placement(node_pos_dp.to(device), density_map_layer, init_density_map, data);
        // logger.info("After DP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f", hpwl.item<float>(),
        //             overflow.item<float>());
        if (true) {
            auto info = make_tuple(-1, 0, data.design_name + "_DP_");
            draw_fig_with_cairo_cpp(node_pos_dp, node_size_dp, data, info);
        }

        node_pos = node_pos_dp;

        torch::Tensor conn_node_pos =
        torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})}, 0);
        torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
            conn_node_pos.to(data.device), data.pin_id2node_id.to(data.device), data.pin_rel_cpos.to(data.device));
        torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach()));
        logger.info("After DP, eval, exact HPWL: %.4E", hpwl.item().toFloat());
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl.item<float>() / (float)data.site_width;
        hpwl_state.hpwls_dp = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];
        hpwl_state.log();
    }

    if (st::setting.eval_params) {

#define EXTRACT(x, a, b, c)         \
  do                                \
  {                                 \
    std::string xs((#x));           \
    std::size_t pos = xs.find("."); \
    xs = xs.substr(pos + 1);        \
    a += (xs + c);                  \
    std::stringstream ss;           \
    ss << (x);                      \
    std::string o;                  \
    ss >> (o);                      \
    b += (o + c);                   \
  } while (false)
        auto [hpwl, overflow] =
            evaluate_placement(node_pos.to(device), density_map_layer, init_density_map, data);
        logger.info("After DP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f", hpwl.item<float>(),
                    overflow.item<float>());
        float overflow_v = overflow.item<float>();
        

        /* dump to csv */
        ifstream infile;
        infile.open("./script_outputs/gp_params_output_log.csv");
        bool is_empty = infile.peek() == std::ifstream::traits_type::eof();

        ofstream outfile;
        outfile.open("./script_outputs/gp_params_output_log.csv", std::ios_base::app);

        string tag = "";
        string value = "";
        string quote = ",";
        string newline = "\n";
        EXTRACT(st::setting.design_name, tag, value, quote);
        EXTRACT(st::setting.quad_penalty, tag, value, quote);
        EXTRACT(st::setting.quad_coeff, tag, value, quote);
        EXTRACT(st::setting.wa_coeff, tag, value, quote);
        EXTRACT(st::setting.density_weight, tag, value, quote);
        EXTRACT(st::setting.density_weight_coef, tag, value, quote);
        EXTRACT(st::setting.magic_hpwl, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_gp, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_lg, tag, value, quote);
        EXTRACT(overflow_v, tag, value, newline);


        cout << tag << endl;
        cout << value << endl;

        if (is_empty) {
            outfile << tag;
        }
        outfile << value;
        outfile.close();

        exit(1);
    }
}


// def commit_node_pos_to_gpdb(node_pos, gpdb, data: PlaceData):
//     exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift)
//     exact_node_lpos = torch.round(exact_node_pos - torch.round(data.node_size * data.die_scale) / 2).cpu()
//     gpdb.apply_node_lpos(exact_node_lpos)