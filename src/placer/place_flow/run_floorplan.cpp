#include "../run_placement.h"

torch::Tensor run_floorplan(NodeData& data,
                            torch::Tensor node_pos,
                            states& hpwl_state,
                            int& cell_mov_lhs,
                            int& cell_mov_rhs) {
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    auto node_die = data.node_die.clone();
    auto device = data.device;
    /* cell visualization information */
    torch::Tensor node_size_bot =
        data.node_size_bot * (1 - node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
    torch::Tensor node_size_top = data.node_size_top * node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);

    torch::Tensor node_pos_lg = node_pos.index({Slice({cell_mov_lhs, cell_mov_rhs})});
    torch::Tensor node_size_lg = data.node_size.clone().to(torch::kCPU);
    node_size_lg.index({Slice(cell_mov_rhs, None)}) -= data.bondingInfo[2].to(torch::kCPU);  // FIXME: w+space -> w
    //dp::DetailedPlaceDataTensor lg_db_at(data, node_pos_lg, node_size_lg);
    
    if(dp::macroFloorplan(data,
                        node_pos_lg,
                        data.mov_node_weights,
                        data.numRows,
                        data.rowHeights,
                        0,
                        cell_mov_lhs,
                        cell_mov_rhs) == false)
    {
        run_greedy_place_for_fp(data);
        node_pos_lg = data.node_pos;
    }
    node_die = data.node_die.clone();

    /*
    for (int i = 0; i < 2; i++) {
        logger.info("============= Chip-%d-Cell FP =============", i);
        //update node_weight for each chip
        torch::Tensor node_weight = data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(torch::kCPU);
        //node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));
        node_weight = torch::_cast_Int(node_weight);

        lg_db_at.update_node_weight(node_weight);
        dp::macroFloorplan(data,
                           lg_db_at,
                           node_pos_lg,
                           data.numRows[i].item<int>(),
                           data.rowHeights[i].item<float>(),
                           0,
                           cell_mov_lhs,
                           cell_mov_rhs);
        auto [hpwl1, hpwl2, tmp] =
            evaluate_wl_cross_chip(node_pos_lg.to(data.device), data.node_die.to(data.device), data);
        logger.info("After Floorplan, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
    }
    */

    /* evalutate wirelength */
    auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos_lg.to(device), data.node_die.to(device), data);
    auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
        evaluate_wl_ovhd_cross_chip(node_pos_lg.to(device), data.node_die.to(device), data);
    // logger.notice("After Cell LG, exact HPWL [bot + top = total / overlap / overhead]: [%d + %d = %d / %d / %d]",
    // hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
    // hpwl_ovhd.item<int>());
    logger.notice(
        "After FP, exact HPWL [bot + top = total / overlap / overhead]: [%.2f + %.2f = %.2f / %.2f / %.2f]",
        hpwl1.item<float>(),
        hpwl2.item<float>(),
        (hpwl1 + hpwl2).item<float>(),
        hpwl_ovlp.item<float>(),
        hpwl_ovhd.item<float>());
    // hpwl_state.hpwls_lg = (hpwl1 + hpwl2).item<int>();
    hpwl_state.hpwls_lg = (hpwl1 + hpwl2).item<float>();
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl_state.hpwls_lg;
    hpwl_state.log();
    /* draw cells */
    if (true) {
        torch::Tensor node_size_top = 
            data.node_size_top * node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
        torch::Tensor node_size_bot =
            data.node_size_bot * (1 - node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
        auto cell_node_pos_lg = node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_FP_0");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_FP_1");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_FP_2");
        auto node_pos_draw_cp = torch::cat({cell_node_pos_lg, cell_node_pos_lg}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
    }

    node_pos = node_pos_lg;
    /* save to .pt model */
    if (st::setting.save_model) {
        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string model_dir = res_root.string() + "/lg.pt";
        logger.info("Save model to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
    }

    printlog(LOG_INFO, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    return node_pos;
}