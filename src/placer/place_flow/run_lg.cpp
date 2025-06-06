#include "../run_placement.h"

torch::Tensor run_lg(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state, int& cell_mov_lhs,
                     int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs, int& mov_rhs, bool only_macro, bool only_cells, bool only_vias) {
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    auto node_die = data.node_die.clone();
    auto device = data.device;
    /* cell visualization information */
    torch::Tensor node_size_bot =
        data.node_size_bot * (1 - node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
    torch::Tensor node_size_top = data.node_size_top * node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);

    torch::Tensor via_node_pos_lg = node_pos.index({Slice(cell_mov_rhs, mov_rhs)});
    torch::Tensor via_node_pos_init = via_node_pos_lg.clone();
    torch::Tensor via_node_size_lg = via_data.node_size;
    if (!only_macro && st::setting.via_dp && only_vias) {
        torch::Tensor via_node_weight = via_data.bonding_map.to(torch::kCPU);

        torch::Tensor node_pos_lg = node_pos;
        torch::Tensor node_size_lg = torch::cat({torch::zeros({data.num_nodes, 2}), via_node_size_lg}, 0);

        dp::DetailedPlaceDataTensor lg_db_at(data, node_pos_lg, node_size_lg);
        // if (false) {
        if (st::setting.lg_ver == 2) {
            logger.info("======================= VIA Legalization =======================");
            /* update node_weight for each chip */
            torch::Tensor node_weight = torch::zeros({data.num_nodes}, dtype(torch::kInt));
            node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));

            lg_db_at.update_node_weight(node_weight);

            dp::DetailedPlaceData lg_db(data, lg_db_at, via_data.numRows.item<int>(),
                                        via_data.row_height.item<float>());
            lg_db.num_movable_nodes = lg_db.num_nodes;
            lg_db.xl = via_data.core_info[0].item<float>();
            lg_db.xh = via_data.core_info[1].item<float>();
            lg_db.yl = via_data.core_info[2].item<float>();
            lg_db.yh = via_data.core_info[3].item<float>();
            if (st::setting.sw) lg_db.site_width = 
                        (via_data.bondingInfo[0] + via_data.bondingInfo[2]).item<float>();

            dp::greedyLegalizationV2(lg_db, 1, 64);
            dp::abacusLegalizationV2(lg_db, 1, 64);
            lg_db_at.update_node_pos(node_pos_lg);

            node_pos = node_pos_lg;

        } else {
            logger.info("======================= VIA Legalization =======================");
            /* Via Legalization */
            /* Greedy Legalization */
            greedyLegalization(via_node_pos_init, via_node_size_lg, via_node_pos_lg, via_node_weight,
                               via_data.core_info, via_data.numRows, via_data.row_height, 1, 64,
                               via_node_pos_init.size(0));
            /* Abacus Legalization */
            abacusLegalization(via_node_pos_init, via_node_size_lg, via_node_pos_lg, via_node_weight,
                               via_data.core_info, via_data.numRows, via_data.row_height, 1, 64,
                               via_node_pos_init.size(0));
            node_pos_lg = torch::cat({node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_node_pos_lg}, 0);
            node_pos = node_pos_lg;
        }

        auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
            evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        logger.notice("After VIA LG, exact HPWL [bot + top = total / overlap / overhead]: [%f + %f = %f / %f / %f]",
                      hpwl1.item<float>(), hpwl2.item<float>(), (hpwl1 + hpwl2).item<float>(), hpwl_ovlp.item<float>(),
                      hpwl_ovhd.item<float>());
    }

    torch::Tensor node_pos_lg = node_pos;
    torch::Tensor node_size_lg = data.node_size.clone().to(torch::kCPU);
    node_size_lg.index({Slice(cell_mov_rhs, None)}) -= data.bondingInfo[2].to(torch::kCPU);  // FIXME: w+space -> w
    dp::DetailedPlaceDataTensor lg_db_at(data, node_pos_lg.to(torch::kCPU), node_size_lg);
    
    if (only_cells) {
        if (st::setting.lg_ver == 2) {
            logger.info("============= Legalizer-ver.%d =============", st::setting.lg_ver);
            for (int i = 0; i < 2; i++) {
                logger.info("============= Chip-%d-Cell LG =============", i);
                /* update node_weight for each chip */
                torch::Tensor node_weight =
                    data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(torch::kCPU);
                torch::Tensor via_node_weight = via_data.bonding_map.clone().to(torch::kCPU) * 0;
                node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));

                lg_db_at.update_node_weight(node_weight);
                dp::legalizationV2(data, lg_db_at, node_pos_lg, data.numRows[i].item<int>(),
                                data.rowHeights[i].item<float>(), 0, 1, 64,cell_mov_lhs, cell_mov_rhs, only_macro);
                auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_lg.to(data.device), data.node_die.to(data.device), data);
                logger.info("After Greedy-Abacus LG, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                            hpwl1.item<float>(),
                            hpwl2.item<float>(),
                            (hpwl1 + hpwl2).item<float>());
            }
        } else {
            logger.info("============= Legalizer-ver.%d =============", st::setting.lg_ver);
            dp::legalizationV1(data, node_pos_lg, node_size_lg, cell_mov_lhs, cell_mov_rhs);
        }
    }
    

    /* evalutate wirelength */
    auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
    auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
        evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
    //logger.notice("After Cell LG, exact HPWL [bot + top = total / overlap / overhead]: [%d + %d = %d / %d / %d]",
                  //hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
                  //hpwl_ovhd.item<int>());
    logger.notice("After Cell LG, exact HPWL [bot + top = total / overlap / overhead]: [%.2f + %.2f = %.2f / %.2f / %.2f]",
                  hpwl1.item<float>(), hpwl2.item<float>(), (hpwl1 + hpwl2).item<float>(), hpwl_ovlp.item<float>(),
                  hpwl_ovhd.item<float>());
    //hpwl_state.hpwls_lg = (hpwl1 + hpwl2).item<int>();
    hpwl_state.hpwls_lg = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl_state.hpwls_lg;
    hpwl_state.hpwls[hpwl_state.hpwl_idx - 2] *= data.site_width;
    hpwl_state.log();

    /* draw cells */
    if (!only_macro) {
        auto cell_node_pos_lg = node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(cell_node_pos_lg);
        cell_node_pos_lg = cell_node_pos_lg + node_shift;
        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_LG_0");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_LG_1");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_LG_2");
        auto node_pos_draw_cp = torch::cat({cell_node_pos_lg, cell_node_pos_lg}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
        /* draw vias */
        auto info = make_tuple(st::setting.round_recursion, 0, data.design_name + "_VIA_LG");
        draw_fig_with_cairo_cpp(via_node_pos_lg, via_node_size_lg, via_data, info, viaColor);
    }

    node_pos = node_pos_lg;
    /* save to .pt model */
    if (!only_macro && st::setting.save_model) {
        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string model_dir = res_root.string() + "/lg.pt";
        logger.info("Save model to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
    }

    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    return node_pos;
}