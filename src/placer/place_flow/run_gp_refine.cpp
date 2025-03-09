#include "../run_placement.h"

torch::Tensor run_gp_refine(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state,
                            int& cell_mov_lhs, int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs,
                            int& mov_rhs) {
    logger.info("=========== refine placement ============");

    st::setting.num_den_layer = 2;
    st::setting.filler_type = "surround";
    st::setting.rf_flag = true;
    st::setting.early_stop_check_plateau = false;
    st::setting.num_bin_x = 512;
    st::setting.num_bin_y = 2048;
    st::setting.block_row = 1;

    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    auto node_die = data.node_die.clone();
    auto device = data.device;
    /* cell visualization information */
    torch::Tensor node_size_bot = data.node_size_bot * (1 - node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)})).unsqueeze(1);
    torch::Tensor node_size_top = data.node_size_top * node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
    // data.reset();

    /* movable cells */
    // torch::Tensor init_density_map = get_init_density_map(data);
    torch::Tensor init_density_maps = get_init_density_map_cross_chip(data);
    auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info_cross_chip();

    // torch::Tensor via_init_density_map =
    //     torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);
    // via_data.init_vars();
    // std::tie(via_mov_lhs, via_mov_rhs) = via_data.movable_index;
    // auto [via_mov_node_pos, via_mov_node_size, via_expand_ratio] = via_data.get_mov_node_info();

    via_data.node_pos = node_pos.index({Slice(cell_mov_rhs, None)}).clone();

    // /* movable cells&vias*/
    // auto [mov_node_pos_all, mov_node_size_all, expand_ratio_all] =
    //     data.get_mov_node_info_cross_chip_with_via(via_data, make_tuple(mov_node_pos, mov_node_size, expand_ratio),
    //                                                make_tuple(via_mov_node_pos, via_mov_node_size,
    //                                                via_expand_ratio));
    // std::tie(mov_lhs, mov_rhs) = data.movable_index;

    // data.get_mov_node_info_cross_chip();

    logger.info("Node type: %d-%d-%d", mov_lhs, cell_mov_rhs, mov_rhs);
    data.fixed_connected_index = make_tuple(cell_mov_rhs, mov_rhs);
    std::tie(mov_lhs, mov_rhs) = data.movable_index;
    logger.info("Mov cell: %d-%d", mov_lhs, mov_rhs);

    logger.info("load from prev sol of size %d...", node_pos.size(0));
    mov_node_pos.index({Slice(cell_mov_rhs, mov_rhs), Slice(0, 2)})
        .data()
        .copy_(node_pos.index({Slice(cell_mov_rhs, mov_rhs), Slice(0, 2)}).data());

    // // for (int i = 0; i < mov_node_pos.size(0); i++) {
    // //     mov_node_pos[i][0] = node_pos[i][0].item<float>();
    // //     mov_node_pos[i][1] = node_pos[i][1].item<float>();
    // // }
    // for (int i = cell_mov_rhs; i < mov_rhs; i++) {
    //     mov_node_pos[i][0] = node_pos[i][0].item<float>();
    //     mov_node_pos[i][1] = node_pos[i][1].item<float>();
    // }

    mov_node_pos = mov_node_pos.to(data.device).requires_grad_(true);
    mov_node_size = mov_node_size.to(data.device);
    expand_ratio = expand_ratio.to(data.device);
    data.to(data.device);

    /* trunc nodes to core */
    torch::Tensor node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4;
    torch::Tensor node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4;

    std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn = [&node_pos_lb, &node_pos_ub](torch::Tensor x) {
        x.data().clamp_(node_pos_lb, node_pos_ub);
        return x;
    };

    /* overflow function for each electrostatic field */
    vector<ElectronicDensityLayer> density_map_layers;  // TODO:
    for (int i = 0; i < 2; i++) {
        std::function<torch::Tensor(torch::Tensor)> overflow_fn_cc = [&data, i](torch::Tensor mov_density_map) {
            torch::Tensor overflow_sum = ((mov_density_map - 1) * data.bin_area).clamp_(0.0).sum();
            return overflow_sum / data.mov_cell_areas[i];
        };
        auto overflow_helper_cc = make_tuple(mov_lhs, mov_rhs, overflow_fn_cc);
        density_map_layers.emplace_back(data.unit_len, data.num_bin_x, data.num_bin_y, device, overflow_helper_cc,
                                        expand_ratio, data.sorted_maps, data.macro_mask);
    }

    /* connected fixed Via */
    torch::Tensor conn_fix_node_pos = node_pos.new_empty({0, 2});
    if (get<0>(data.fixed_connected_index) < get<1>(data.fixed_connected_index)) {
        auto [lhs, rhs] = data.fixed_connected_index;
        logger.info("ConnFix type: %d-%d", lhs, rhs);
        conn_fix_node_pos = node_pos.index({Slice(lhs, rhs), "..."}).to(device);
    }

    for (int c_id = 0; c_id < 2; c_id++) {
        torch::Tensor node_weight = data.mov_node_weights[c_id];
        auto den_val_list =
            density_map_layers[c_id].forward(mov_node_pos, mov_node_size, init_density_maps[c_id], node_weight, false);
    }

    for (int c_id = 0; c_id < 2; c_id++) {
        /* parameteer scheduler */
        ParamScheduler ps = ParamScheduler(data);

        /* objective function */
        /* follow the order of node types */
        /* | MovConnected | MovFloat | ConnFixed ... | */
        conn_fix_node_pos = conn_fix_node_pos.detach();
        std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> obj_and_grad_fn =
            [&trunc_node_pos_fn, &mov_node_size, &init_density_maps, &density_map_layers, &conn_fix_node_pos, &ps, &data,
             &c_id](at::Tensor mov_node_pos) {
                return calc_obj_and_grad_with_via(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_maps,
                                                  density_map_layers, conn_fix_node_pos, ps, data, c_id);
            };

        /* evaluation function */
        std::function<tuple<torch::Tensor, torch::Tensor, torch::Tensor>(torch::Tensor)> evaluator_fn =
            [&trunc_node_pos_fn, &mov_node_size, &init_density_maps, &density_map_layers, &conn_fix_node_pos, &ps,
             &data](at::Tensor mov_node_pos) {
                return fast_evaluator_multi_circuit(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_maps,
                                                    density_map_layers, conn_fix_node_pos, ps, data);
            };

        /* Nesterov optimizer */
        auto optimizer = torch::optim::Nesterov({mov_node_pos}, torch::optim::NesterovOptions(0.0), obj_and_grad_fn);
        init_params_with_via(mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, density_map_layers,
                             mov_node_size, init_density_maps, optimizer, ps, data, c_id);

        /* learning rate */
        double init_lr =
            estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, st::setting.lr);
        logger.info("Init learning rate %.3E", init_lr);
        for (auto& group : optimizer.param_groups()) {
            auto& options = static_cast<torch::optim::NesterovOptions&>(group.options());
            options.set_lr(init_lr);
        }

        /* start gp iteration */
        if (true) {
            auto [hpwls, overflows, tmp] = evaluator_fn(mov_node_pos);
            logger.info(
                "iter: %d | masked_hpwl: (%.2E, %.2E) overflow: (%.4f, %.4f) "
                "density_weight: %.4E wa_coeff: %.4E",
                0, hpwls[0].item<float>(), hpwls[1].item<float>(), overflows[0].item<float>(),
                overflows[1].item<float>(), ps.density_weight, ps.wa_coeff);
        }
        printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(),
                 utils::mem_use::get_peak());
        logger.info("=========================================");
        logger.info("start gp for chip-%d", c_id);
        int& iteration = st::setting.iteration;
        iteration = 0;  // FIXME: 0 ? 1
        for (iteration = 0; iteration < st::setting.inner_iter; iteration++) {
            torch::Tensor obj = optimizer.step();

            auto [hpwls, overflows, tmp] = evaluator_fn(mov_node_pos);
            ps.step(hpwls[c_id].item<float>(), overflows[c_id].item<float>(), mov_node_pos);

            if (iteration % st::setting.log_freq == 0 || iteration == st::setting.inner_iter - 1 ||
                ps.need_to_early_stop()) {
                logger.info(
                    "iter: %d | masked_hpwl: (%.2E, %.2E) overflow: (%.4f, %.4f) obj: %.4E "
                    "density_weight: %.4E wa_coeff: %.4E",
                    iteration, hpwls[0].item<float>(), hpwls[1].item<float>(), overflows[0].item<float>(),
                    overflows[1].item<float>(), obj.item<float>(), ps.density_weight, ps.wa_coeff);
                if (true) {
                    auto info1 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_GP_RF_0");
                    draw_fig_with_cairo_cpp(
                        mov_node_pos.to(torch::kCPU).index({Slice(cell_mov_lhs, cell_mov_rhs), "..."}), node_size_bot,
                        data, info1);
                    auto info2 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_GP_RF_1");
                    draw_fig_with_cairo_cpp(
                        mov_node_pos.to(torch::kCPU).index({Slice(cell_mov_lhs, cell_mov_rhs), "..."}), node_size_top,
                        data, info2);
                    /* draw vias */
                    auto info = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_VIA_RF");
                    draw_fig_with_cairo_cpp(via_data.node_pos, via_data.node_size, via_data, info, viaColor);
                }
            }
            if (ps.need_to_early_stop()) {
                break;
            }
        }
        printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(),
                 utils::mem_use::get_peak());
        /* retrieve best score and evaluate without filler */
        auto [best_sol, best_hpwl, best_overflow, best_iteration] = ps.get_best_solution();
        logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f", iteration, best_hpwl, best_overflow);
        if (best_sol.numel() != 0) {
            mov_node_pos.data().copy_(best_sol);
        }
    }

    data.to(torch::kCPU);
    via_data.to(torch::kCPU);

    auto conn_node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs)});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(conn_node_pos.to(device), data.node_die.to(device), data);
    logger.info("After GP Refinement, exact HPWL of two chips: (bot + top = total): (%d + %d = %d)", hpwl1.item<int>(),
                hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
    hpwl_state.hpwls_gp = (hpwl1 + hpwl2).item<int>();
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl_state.hpwls_gp;

    if(st::setting.block_row) {
        data.node_size.index({"...", 1})     /= 0.9; // TODO: config
        data.node_size_top.index({"...", 1}) /= 0.9;
        data.node_size_bot.index({"...", 1}) /= 0.9;
        data.node_size = torch::cat({data.node_size.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_data.node_size}, 0);  // FIXME: original size before expand
    }
    auto node_size = data.node_size.index({Slice(mov_lhs, mov_rhs)}).clone();
    

    mov_node_pos = mov_node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(torch::kCPU);
    if (true) {
        /* draw cells */
        auto info1 = make_tuple(st::setting.round_recursion, -5, data.design_name + "_GP_RF_0");
        draw_fig_with_cairo_cpp(mov_node_pos, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, -5, data.design_name + "_GP_RF_1");
        draw_fig_with_cairo_cpp(mov_node_pos, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, -5, data.design_name + "_GP_RF_2");
        auto node_pos_draw_cp = torch::cat({mov_node_pos, mov_node_pos}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);

        auto info = make_tuple(st::setting.round_recursion, -5, data.design_name + "_VIA_RF");
        draw_fig_with_cairo_cpp(conn_fix_node_pos.to(torch::kCPU), via_data.node_size, via_data, info, viaColor);
    }

    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    st::setting.round_recursion--;
    return conn_node_pos.to(torch::kCPU);
}