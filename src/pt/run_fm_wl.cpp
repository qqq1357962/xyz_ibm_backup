
#include "partition.h"
#include "patoh.h"

void Partitioner::run_fm_wl(NodeData& data, bool skip) {
    // ======================================================================================================
    //
    //                                            PARTITION
    //
    // ======================================================================================================
    logger.info("=================== Running fm-wl ===================");
    // auto node_pos = data.node_pos;
    auto node_pos = node_pos_2d_ground;
    auto node_size = data.node_size;
    macro_mask = data.macro_mask.clone();
    Myreg_mask = data.Myreg_mask.clone();
    // node_die = data.node_die;

    vector<Macro_Box> Macro_Boxs;
    vector<Macro_Box> Small_Macro_Boxs;
    float shrink_ratio = 0.25;
    auto node_pos_a = node_pos.accessor<float, 2>();
    auto node_size_a = node_size.accessor<float, 2>();
    auto macro_mask_a = macro_mask.accessor<float, 1>();
    auto Myreg_mask_a = Myreg_mask.accessor<float, 1>();
    auto non_zero_indices = torch::nonzero(macro_mask);
    auto non_zero_num = macro_mask.sum().item<int>();
    auto node_die_a = node_die.accessor<int, 1>();
    for (int i = 0; i < non_zero_num; i++) {
        int c_id = node_die_a[non_zero_indices[i].item<int>()];
        Macro_Boxs.emplace_back(
            node_pos_a[non_zero_indices[i].item<int>()][0] - node_size_a[non_zero_indices[i].item<int>()][0] / 2,
            node_pos_a[non_zero_indices[i].item<int>()][1] - node_size_a[non_zero_indices[i].item<int>()][1] / 2,
            node_pos_a[non_zero_indices[i].item<int>()][0] + node_size_a[non_zero_indices[i].item<int>()][0] / 2,
            node_pos_a[non_zero_indices[i].item<int>()][1] + node_size_a[non_zero_indices[i].item<int>()][1] / 2, 
            c_id);

        float shrink_size =
            min(node_size_a[non_zero_indices[i].item<int>()][0], node_size_a[non_zero_indices[i].item<int>()][1]) *
            shrink_ratio;
        Small_Macro_Boxs.emplace_back(node_pos_a[non_zero_indices[i].item<int>()][0] -
                                          (node_size_a[non_zero_indices[i].item<int>()][0] - shrink_size) / 2,
                                      node_pos_a[non_zero_indices[i].item<int>()][1] -
                                          (node_size_a[non_zero_indices[i].item<int>()][1] - shrink_size) / 2,
                                      node_pos_a[non_zero_indices[i].item<int>()][0] +
                                          (node_size_a[non_zero_indices[i].item<int>()][0] - shrink_size) / 2,
                                      node_pos_a[non_zero_indices[i].item<int>()][1] +
                                          (node_size_a[non_zero_indices[i].item<int>()][1] - shrink_size) / 2,
                                      c_id);

    }

    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < num_nodes; ++i) {
        int c_id = node_die[i].item<int>();
        nodes[i]->group = c_id;
        mov_cell_areas[c_id] += nodes[i]->sizes[c_id];
    }

    for (int i = 0; i < num_nodes; ++i) {
        if(Myreg_mask_a[i] == 1)
        {
            assert(node_die_a[i] == 1);
        }
        if (macro_mask_a[i] != 1) {
            int c_id = node_die_a[i];
            int other_c_id = 1 - c_id;
            bool self_overlap = false;
            bool oppo_overlap = false;
            for (auto j : Macro_Boxs) {
                bool inside_macro = j.comp(node_pos_a[i][0] - node_size_a[i][0] / 2,
                                           node_pos_a[i][1] - node_size_a[i][1] / 2,
                                           node_pos_a[i][0] + node_size_a[i][0] / 2,
                                           node_pos_a[i][1] + node_size_a[i][1] / 2,
                                           other_c_id);
                if (inside_macro) {
                    macro_mask_a[i] = 1;
                    oppo_overlap = true;
                    break;
                }
            }
            if (!st::setting.fm_cut_size) {
                for (auto j : Small_Macro_Boxs) {
                    bool inside_macro = j.comp(node_pos_a[i][0] - node_size_a[i][0] / 2,
                                            node_pos_a[i][1] - node_size_a[i][1] / 2,
                                            node_pos_a[i][0] + node_size_a[i][0] / 2,
                                            node_pos_a[i][1] + node_size_a[i][1] / 2,
                                            c_id);
                    if (inside_macro) {
                        // macro_mask_a[i] = 1;
                        self_overlap = true;
                        break;
                    }
                }
                if (self_overlap && !oppo_overlap) {
                    node_die[i] = 1 - node_die[i];
                    data.node_die[i] = 1 - data.node_die[i];
                    nodes[i]->group = other_c_id;
                    mov_cell_areas[c_id] -= nodes[i]->sizes[c_id];
                    mov_cell_areas[other_c_id] += nodes[i]->sizes[other_c_id];
                }
            }
        }
    }

    /* updata pin rel pos */
    for (int i = 0; i < num_pins; ++i) {
        int64_t node_id = data.pin_id2node_id[i].item<long>();
        if (node_die[node_id].item<int>() == 0) {
            data.pin_rel_cpos[i] = data.pin_rel_cpos_bot[i];
        } else {
            data.pin_rel_cpos[i] = data.pin_rel_cpos_top[i];
        }
    }

    pt::PartitionDataTensor pt_db_at(data, node_pos, node_size);
    pt_db_at_ptr = make_shared<pt::PartitionDataTensor>(pt_db_at);
    pt::PartitionData pt_db(data, pt_db_at);

    /* HPWL metrics */
    auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), node_die.to(device), data);
    data.hyperedge_list = data.hyperedge_list.to(device);
    data.hyperedge_list_end = data.hyperedge_list_end.to(device);
    torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
        node_pos.to(device), data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device));
    torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach())); // FIXME:
    float hpwl_float = (hpwl + hpwl_ovlp).item<float>();
    logger.info("============= Before WL-PT, estimated wl cross-chip (%f + %f = %f) =============",
                hpwl.item<float>(),
                hpwl_ovlp.item<float>(),
                (hpwl + hpwl_ovlp).item<float>());
    logger.info("WL at each die [%f / %f]", hpwl1.item<float>(), hpwl2.item<float>());

    utils::timer runtime;

    /* run FM algorithm */
    // GAIN = 0;
    // int64_t GAIN_dif = INT_MAX;
    // GAINS.push_back(0);
    GAIN_FLOAT = 0;
    float GAIN_dif = INT_MAX;
    GAINS_FLOAT.push_back(0);
    int iteration = 0;
    // running = true;
    running = !skip;
    if (skip)
    {
        logger.warning("Skip HPWL-FM iterations.");
    }
    int max_iters = st::setting.fmwl_iter;
    std::vector<float> hpwls(max_iters + 1);
    hpwls[0] = hpwl_float;
    hpwl_track = hpwl_float;
    rpt_cut_size();
    //for (int iteration = 0; iteration < max_iters && running; iteration++) {
    for (int iteration = 0; iteration < max_iters && running; iteration++) {
        initList(iteration != 0);
        initWLGain(pt_db, false);
        passWL(pt_db);
        // pass();
        GAIN_dif = GAIN_FLOAT - GAINS_FLOAT.back();
        GAINS_FLOAT.push_back(GAIN_FLOAT);
        tracker.clear();
        /* HPWL metrics */
        node_die = pt_db_at.node_die;
        std::tie(hpwl1, hpwl2, hpwl_ovlp) =
            evaluate_wl_cross_chip(node_pos.to(data.device), node_die.to(data.device), data);
        data.hyperedge_list = data.hyperedge_list.to(device);
        data.hyperedge_list_end = data.hyperedge_list_end.to(device);
        pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
            node_pos.to(device), data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device));
        hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach())); // FIXME:
        data.hyperedge_list = data.hyperedge_list.to(torch::kCPU);
        data.hyperedge_list_end = data.hyperedge_list_end.to(torch::kCPU);

        hpwl_float = (hpwl + hpwl_ovlp).item<float>();
        hpwls[iteration + 1] = hpwl_float;

        logger.info("============ iteration %d: hpwl %.3f => %.3f (imp. %g%%) ============",
                    iteration,
                    hpwls[0],
                    hpwls[iteration + 1],
                    (1.0 - hpwls[iteration + 1] / (double)hpwls[0]) * 100);

        logger.info("%d cells change their layer", maxGAINIndex + 1);
        if (GAIN_dif < 1E3) break;
    }
    rpt_cut_size();

    std::tie(hpwl1, hpwl2, hpwl_ovlp) =
        evaluate_wl_cross_chip(node_pos.to(data.device), node_die.to(data.device), data);
    data.hyperedge_list = data.hyperedge_list.to(device);
    data.hyperedge_list_end = data.hyperedge_list_end.to(device);
    pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
        node_pos.to(device), data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device));
    hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach())); // FIXME:
    data.hyperedge_list = data.hyperedge_list.to(torch::kCPU);
    data.hyperedge_list_end = data.hyperedge_list_end.to(torch::kCPU);

    /* HPWL metrics */
    logger.info("============= After WL-PT estimated wl cross-chip (%f + %f = %f) =============",
                hpwl.item<float>(),
                hpwl_ovlp.item<float>(),
                (hpwl + hpwl_ovlp).item<float>());
    logger.info("WL at each die [%f / %f]", hpwl1.item<float>(), hpwl2.item<float>());
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = (hpwl + hpwl_ovlp).item<float>();
    hpwl_state.log();
    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    data.to(torch::kCPU);

    /* dump runtime */
    double run_time = runtime.elapsed();
    logger.warning("========== Execution time: %.2f s ==========\n", run_time);

    /* process die area*/
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < num_nodes; i++) {
        int group = node_die[i].item<int>();
        nodes[i]->group = group;
        mov_cell_areas[group] += nodes[i]->sizes[group];
    }
    logger.info("============ Original partition result ============");
    rpt_cut_size();
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    // /* validate max-utilization constraints */
    // int forced_moved_stdcell = 0;
    // int forced_moved_macro = 0;
    // // mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    // //比例大的优先去top die，比例小的优先去bottom die
    // //给一个双向指针, 大的检测bottom,不行送去top,, 小的检测top, 
    // float area_bot = mov_cell_areas[0].item<float>();
    // float area_top = mov_cell_areas[1].item<float>();
    // float bound_bot = max_mov_cell_areas[0].item<float>();
    // float bound_top = max_mov_cell_areas[1].item<float>();
    // if(area_bot>bound_bot || area_top>bound_top)
    // {
    //     auto ratio_nodes = data.node_area_bot.to(torch::kFloat32)/data.node_area_top.to(torch::kFloat32);
    //     auto [sorted_tensor, indices] = ratio_nodes.sort();
    //     int p_top = 0;
    //     int p_bot = num_nodes-1;
    //     int force_moved_cnt = 0;
    //     int bottom_to_top_cnt = 0;
    //     int top_to_bottom_cnt = 0;
    //     int move_macro=false;
    //     for (int i = 0; i < num_nodes; i++) {
    //         if(area_bot<=bound_bot && area_top<=bound_top)
    //         {
    //             logger.info("in round %d, area is finally legalized! %d cells are moved in validation step", i, force_moved_cnt);
    //             logger.info("during the process, %d cells are moved from top to bottom, %d cells are moved from bottom to top", 
    //                         top_to_bottom_cnt, bottom_to_top_cnt);
    //             break;
    //         }
    //         if(area_top>bound_top)
    //         {
    //             int node_id = indices[p_top].item<int>();
    //             if(data.macro_mask[node_id].item<int>()==1&&(!move_macro))
    //             {
    //             }
    //             else if(node_die[node_id].item<int>()==1)
    //             {
    //                 node_die[node_id]=0;
    //                 area_top-=nodes[node_id]->sizes[1];
    //                 area_bot+=nodes[node_id]->sizes[0];
    //                 force_moved_cnt++;
    //                 top_to_bottom_cnt++;
    //             }
    //             p_top++;
    //         }
    //         if(area_bot>bound_bot)
    //         {
    //             int node_id = indices[p_bot].item<int>();
    //             if(data.macro_mask[node_id].item<int>()==1&&(!move_macro))
    //             {
    //             }
    //             else if(node_die[node_id].item<int>()==0)
    //             {
    //                 node_die[node_id]=1;
    //                 area_top+=nodes[node_id]->sizes[1];
    //                 area_bot-=nodes[node_id]->sizes[0];
    //                 force_moved_cnt++;
    //                 bottom_to_top_cnt++;
    //             }
    //             p_bot--;
    //         }
    //     }
    //     if(area_bot<=bound_bot && area_top<=bound_top)
    //     {
    //         logger.info("can not legalize!");
    //         logger.info("during the process, %d cells are moved from top to bottom, %d cells are moved from bottom to top", 
    //                 top_to_bottom_cnt, bottom_to_top_cnt);
    //     }
    // }else{
    //     logger.info("in force legalization step, already legal!");
    // }
    // logger.info("after validation, top: %f/%f, bot: %f/%f",area_top, bound_top,area_bot, bound_bot);
    

    
    // // for (int i = 0; i < num_nodes; i++) {
    // //     int group = node_die[i].item<int>();
    // //     if (st::setting.clamp_util) {
    // //         if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
    // //         {
    // //             if(data.macro_mask[i].item<int>()==1)
    // //             {
    // //                 logger.info("moving macro %d from %d to %d", i, group, !group);
    // //                 forced_moved_macro++;
    // //             }else{
    // //                 forced_moved_stdcell++;
    // //             }
    // //             group = !group;
    // //         }
    // //     }
    // //     mov_cell_areas[group] += nodes[i]->sizes[group];
    // //     nodes[i]->group = group;
    // //     node_die[i] = group;
    // // }
    // // logger.info("%d std cells and %d macros are moved in validation step after wl-fm", forced_moved_stdcell, forced_moved_macro);

    // logger.info("============ Legalized partition result ============");
    // rpt_cut_size();
    // logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    // logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    // logger.info("Utils for each chip (%.2f, %.2f)",
    //             (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
    //             (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());
    // data.node_die = node_die;
    
    if (true) {
        int count = 0;
        for (int i = 0 ;i < data.num_nodes; i++) 
            if (data.aspect_ratio[i].item<float>() > 6) 
                if (node_die[i].item<int>() == 1) count++;   
        logger.info("[%d / %d] long cells at chip [0/1]", torch::_cast_Float(data.aspect_ratio > 6).sum().item<int>() - count, count);
    }

    /* visualize partition */
    if (true) {
        torch::Tensor node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
        torch::Tensor node_size_top = data.node_size_top * node_die.unsqueeze(1);
        auto cell_node_pos = node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)});
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)}))
                              .expand_as(cell_node_pos);
        cell_node_pos = cell_node_pos + node_shift;
        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_PT_FMWL_0");
        draw_fig_with_cairo_cpp(cell_node_pos, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_PT_FMWL_1");
        draw_fig_with_cairo_cpp(cell_node_pos, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_PT_FMWL_2");
        auto node_pos_draw_cp = torch::cat({cell_node_pos, cell_node_pos}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
    }
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::initWLGain(pt::PartitionData& db, bool update) {
    freecells = torch::ones(num_nodes, dtype(torch::kBool));
    gainlist = torch::zeros(num_nodes, dtype(torch::kFloat));
    density_gainlist = torch::zeros(num_nodes, dtype(torch::kFloat));
    mov_node_xl = torch::zeros({2, num_nodes}, dtype(torch::kFloat));
    mov_node_xh = torch::zeros({2, num_nodes}, dtype(torch::kFloat));
    mov_node_yl = torch::zeros({2, num_nodes}, dtype(torch::kFloat));
    mov_node_yh = torch::zeros({2, num_nodes}, dtype(torch::kFloat));
    mov_node_xl_b = torch::zeros({2, num_nodes}, dtype(torch::kInt));
    mov_node_yl_b = torch::zeros({2, num_nodes}, dtype(torch::kInt));
    mov_node_xh_b = torch::zeros({2, num_nodes}, dtype(torch::kInt));
    mov_node_yh_b = torch::zeros({2, num_nodes}, dtype(torch::kInt));
    num_x_bin = st::setting.num_bin_x;
    num_y_bin = st::setting.num_bin_y;
    bin2node_id.resize(num_x_bin * num_y_bin);
    float min_xl = std::numeric_limits<float>::max();
    float max_xh = -std::numeric_limits<float>::max();
    float min_yl = std::numeric_limits<float>::max();
    float max_yh = -std::numeric_limits<float>::max();
    for (int i = 0; i < num_nodes; i++) {
        for (int n = 0; n < 2; n++) {
            min_xl = min(min_xl, db.x[i] - db.node_size_xs[n][i] / 2);
            max_xh = max(max_xh, db.x[i] + db.node_size_xs[n][i] / 2);
            min_yl = min(min_yl, db.y[i] - db.node_size_ys[n][i] / 2);
            max_yh = max(max_yh, db.y[i] + db.node_size_ys[n][i] / 2);
        }
    }
    max_xh += 1;
    max_yh += 1;
    unit_len_x = (max_xh - min_xl) / num_x_bin;
    unit_len_y = (max_yh - min_yl) / num_y_bin;
    density_map = torch::zeros({2, num_x_bin, num_y_bin}, dtype(torch::kFloat));
    auto density_map_a = density_map.accessor<float, 3>();

    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        int c_id = db.node_die[node_id];
        float node_xl = (db.x[node_id] - db.node_size_xs[c_id][node_id] / 2 - min_xl) / unit_len_x;
        float node_xh = (db.x[node_id] + db.node_size_xs[c_id][node_id] / 2 - min_xl) / unit_len_x;
        float node_yl = (db.y[node_id] - db.node_size_ys[c_id][node_id] / 2 - min_yl) / unit_len_y;
        float node_yh = (db.y[node_id] + db.node_size_ys[c_id][node_id] / 2 - min_yl) / unit_len_y;
        int node_xl_b = static_cast<int>(std::floor(node_xl));
        int node_yl_b = static_cast<int>(std::floor(node_yl));
        int node_xh_b = static_cast<int>(std::floor(node_xh));
        int node_yh_b = static_cast<int>(std::floor(node_yh));

        for (int j = node_xl_b; j < node_xh_b + 1; j++) {
            float bin_x_l = static_cast<float>(j);
            float overlap_x = overlap(node_xl, node_xh, bin_x_l);
            for (int k = node_yl_b; k < node_yh_b + 1; k++) {
                float bin_y_l = static_cast<float>(k);
                float overlap_y = overlap(node_yl, node_yh, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                density_map_a[c_id][j][k] += overlap_area;
            }
        }
    }

    auto non_zero_indices = torch::nonzero(macro_mask);
    auto non_zero_num = macro_mask.sum().item<int>();
    for (int i = 0; i < non_zero_num; i++) {
        gainlist[non_zero_indices[i].item<int>()] = -std::numeric_limits<float>::max();
    }

    /* get bucket list */
    for (int i = 0; i < num_nets; ++i) {
        if (!db.net_mask[i]) continue;
        vector<Box> boxs;
        boxs.emplace_back(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max());
        boxs.emplace_back(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max());

        vector<Box> box2s;
        box2s.emplace_back(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max());
        box2s.emplace_back(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max());

        vector<int> num_node_count(2, 0);
        for (int net2pin_id = db.flat_net2pin_start_map[i]; net2pin_id < db.flat_net2pin_start_map[i + 1];
             ++net2pin_id) {
            int net_pin_id = db.flat_net2pin_map[net2pin_id];
            int node_id = db.pin2node_map[net_pin_id];
            int c_id = db.node_die[node_id];
            num_node_count[c_id]++;

            float xx = db.x[node_id] + db.pin_offset_xs[c_id][net_pin_id];
            float yy = db.y[node_id] + db.pin_offset_ys[c_id][net_pin_id];

            if (xx > boxs[c_id].xh) {
                box2s[c_id].xh = boxs[c_id].xh;
                boxs[c_id].xh = xx;
            } else if (xx <= boxs[c_id].xh && xx > box2s[c_id].xh)
                box2s[c_id].xh = xx;
            if (yy > boxs[c_id].yh) {
                box2s[c_id].yh = boxs[c_id].yh;
                boxs[c_id].yh = yy;
            } else if (yy <= boxs[c_id].yh && yy > box2s[c_id].yh)
                box2s[c_id].yh = yy;

            if (xx < boxs[c_id].xl) {
                box2s[c_id].xl = boxs[c_id].xl;
                boxs[c_id].xl = xx;
            } else if (xx >= boxs[c_id].xl && xx < box2s[c_id].xl)
                box2s[c_id].xl = xx;
            if (yy < boxs[c_id].yl) {
                box2s[c_id].yl = boxs[c_id].yl;
                boxs[c_id].yl = yy;
            } else if (yy >= boxs[c_id].yl && yy < box2s[c_id].yl)
                box2s[c_id].yl = yy;
        }

        // cout << "--------------------------\n";
        // cout << boxs[0] << endl;
        // cout << boxs[1] << endl;
        // cout << box2s[0] << endl;
        // cout << box2s[1] << endl;

        for (int net2pin_id = db.flat_net2pin_start_map[i]; net2pin_id < db.flat_net2pin_start_map[i + 1];
             ++net2pin_id) {
            int net_pin_id = db.flat_net2pin_map[net2pin_id];
            int node_id = db.pin2node_map[net_pin_id];
            int c_id = db.node_die[node_id];
            int other_c_id = 1 - c_id;
            if (num_node_count[c_id] < 2) continue;

            // cout << "--------------------------\n";
            // cout << "node " << node_id << " at " << c_id << endl;
            // cout << db.x[node_id] << " " << db.pin_offset_xs[c_id][net_pin_id] << " " <<
            // db.pin_offset_xs[other_c_id][net_pin_id] << endl;

            vector<Box> box_swaps;
            box_swaps.emplace_back(boxs[0]);
            box_swaps.emplace_back(boxs[1]);

            float xx = db.x[node_id] + db.pin_offset_xs[c_id][net_pin_id];
            if (xx == boxs[c_id].xh) box_swaps[c_id].xh = box2s[c_id].xh;
            if (xx == boxs[c_id].xl) box_swaps[c_id].xl = box2s[c_id].xl;
            float other_xx = db.x[node_id] + db.pin_offset_xs[other_c_id][net_pin_id];
            if (other_xx > boxs[other_c_id].xh) box_swaps[other_c_id].xh = other_xx;
            if (other_xx < boxs[other_c_id].xl) box_swaps[other_c_id].xl = other_xx;

            float yy = db.y[node_id] + db.pin_offset_ys[c_id][net_pin_id];
            if (yy == boxs[c_id].yh) box_swaps[c_id].yh = box2s[c_id].yh;
            if (yy == boxs[c_id].yl) box_swaps[c_id].yl = box2s[c_id].yl;
            float other_yy = db.y[node_id] + db.pin_offset_ys[other_c_id][net_pin_id];
            if (other_yy > boxs[other_c_id].yh) box_swaps[other_c_id].yh = other_yy;
            if (other_yy < boxs[other_c_id].yl) box_swaps[other_c_id].yl = other_yy;

            // cout << box_swaps[0] << endl;
            // cout << box_swaps[1] << endl;

            // float x_max_mid = min(boxs[0].xh, boxs[1].xh);
            // float x_min_mid = max(boxs[0].xl, boxs[1].xl);
            // float x_ovlp = max(x_max_mid - x_min_mid, (float)0);
            // float x_max_mid_swap = min(box_swaps[0].xh, box_swaps[1].xh);
            // float x_min_mid_swap = max(box_swaps[0].xl, box_swaps[0].xl);
            // float x_ovlp_swap = max(x_max_mid_swap - x_min_mid_swap, (float)0);
            // float gain_x = x_ovlp - x_ovlp_swap;

            // float y_max_mid = min(boxs[0].yh, boxs[1].yh);
            // float y_min_mid = max(boxs[0].yl, boxs[0].yl);
            // float y_ovlp = max(y_max_mid - y_min_mid, (float)0);
            // float y_max_mid_swap = min(box_swaps[0].yh, box_swaps[1].yh);
            // float y_min_mid_swap = max(box_swaps[0].yl, box_swaps[0].yl);
            // float y_ovlp_swap = max(y_max_mid_swap - y_min_mid_swap, (float)0);
            // float gain_y = y_ovlp - y_ovlp_swap;

            // nodes[node_id]->gain_map[i] = (gain_x + gain_y);
            // gainlist[node_id] += (gain_x + gain_y);

            float wl = boxs[0] + boxs[1];
            float wl_swap = box_swaps[0] + box_swaps[1];
            float gain = wl - wl_swap;
            if (macro_mask[node_id].item<int>() == 1) {
                nodes[node_id]->gain_map[i] = -std::numeric_limits<float>::max();
                gainlist[node_id] = -std::numeric_limits<float>::max();
            }
            else if (Myreg_mask[node_id].item<int>() == 1) {
                nodes[node_id]->gain_map[i] = -std::numeric_limits<float>::max();
                gainlist[node_id] = -std::numeric_limits<float>::max();
            }
            else {
                nodes[node_id]->gain_map[i] = gain;
                gainlist[node_id] += gain;
            }
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        if (macro_mask[node_id].item<int>() == 1) {
            density_gainlist[node_id] = -std::numeric_limits<float>::max();
        } else {
            int c_id = db.node_die[node_id];
            float gain_density = 0;
            for (int n = 0; n < 2; n++) {
                int c_id_ = (n == 0) ? c_id : (1 - c_id);
                float node_xl = (db.x[node_id] - db.node_size_xs[c_id_][node_id] / 2 - min_xl) / unit_len_x;
                float node_xh = (db.x[node_id] + db.node_size_xs[c_id_][node_id] / 2 - min_xl) / unit_len_x;
                float node_yl = (db.y[node_id] - db.node_size_ys[c_id_][node_id] / 2 - min_yl) / unit_len_y;
                float node_yh = (db.y[node_id] + db.node_size_ys[c_id_][node_id] / 2 - min_yl) / unit_len_y;
                int node_xl_b = static_cast<int>(std::floor(node_xl));
                int node_yl_b = static_cast<int>(std::floor(node_yl));
                int node_xh_b = static_cast<int>(std::floor(node_xh));
                int node_yh_b = static_cast<int>(std::floor(node_yh));
                mov_node_xl[c_id_][node_id] = node_xl;
                mov_node_xh[c_id_][node_id] = node_xh;
                mov_node_yl[c_id_][node_id] = node_yl;
                mov_node_yh[c_id_][node_id] = node_yh;
                mov_node_xl_b[c_id_][node_id] = node_xl_b;
                mov_node_xh_b[c_id_][node_id] = node_xh_b;
                mov_node_yl_b[c_id_][node_id] = node_yl_b;
                mov_node_yh_b[c_id_][node_id] = node_yh_b;
                float current_density = 0;

                for (int j = node_xl_b; j < node_xh_b + 1; j++) {
                    float bin_x_l = static_cast<float>(j);
                    float overlap_x = overlap(node_xl, node_xh, bin_x_l);
                    for (int k = node_yl_b; k < node_yh_b + 1; k++) {
                        float bin_y_l = static_cast<float>(k);
                        float overlap_y = overlap(node_yl, node_yh, bin_y_l);
                        float overlap_area = overlap_x * overlap_y;
                        current_density += overlap_area * (density_map_a[c_id_][j][k] + ((n == 0) ? 0 : overlap_area));
                        if (bin2node_id[j * num_y_bin + k].size() == 0 ||
                            bin2node_id[j * num_y_bin + k].back() != node_id) {
                            bin2node_id[j * num_y_bin + k].emplace_back(node_id);
                        }
                    }
                }
                gain_density += (n == 0) ? current_density : (-current_density);
            }
            density_gainlist[node_id] = gain_density;
        }
    }
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::passWL(pt::PartitionData& db) {
    int num_free = num_nodes;
    float GAIN_ITER = 0, GAIN_MAX = 0;
    maxGAINIndex = -1;
    int num_swaps = num_nodes / 20;

    float progress = 0.0;
    vector<float> hpwls(num_swaps + 1);
    vector<float> areas(num_swaps + 1);
    vector<float> cuts(num_swaps + 1);
    cuts[0] = cutsize;
    hpwls[0] = hpwl_track;
    areas[0] = (mov_cell_areas[0] / mov_cell_areas[1]).item<float>();
    torch::Tensor hpwls_pass = torch::zeros({num_nodes}, dtype(torch::kFloat));
    torch::Tensor gains_pass = torch::zeros({num_nodes}, dtype(torch::kFloat));
    torch::Tensor cuts_pass = torch::zeros({num_nodes}, dtype(torch::kFloat));
    torch::Tensor areas_pass = torch::zeros({num_nodes}, dtype(torch::kFloat));
    for (int i = 0; i < num_swaps; i++) {
        progress = (float)i / num_swaps;
        if ((i % 100) == 0) {
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }

        auto node_cost_area = torch::exp(
            torch::tensor({(st::setting.fmwl_area_coef * (areas[i] - upper_lower_bound_ratio[0]) /
                            (upper_lower_bound_ratio[1] - upper_lower_bound_ratio[0]))
                               .item<float>(),
                           ((upper_lower_bound_ratio[1] - areas[i]) /
                            (upper_lower_bound_ratio[1] - upper_lower_bound_ratio[0]))
                               .item<float>()},
                          torch::dtype(torch::kFloat)));
        // cout << node_cost_area << endl;
        // auto weight = gainlist.max() / density_gainlist.max() / 2;
        auto cost_list = gainlist.clone() + density_gainlist.clone() * 10;
        if (st::setting.fm_cut_size) {
            cost_list = via_gainlist.clone() + density_gainlist.clone() / 10;
        }
        // cost_list *= node_cost_area.index_select(0, pt_db_at_ptr->node_die);

        // int cell_mov_idx = torch::argmax(gainlist, 0).item<int>();
        int cell_mov_idx = torch::argmax(cost_list, 0).item<int>();

        // int cell_mov_idx = i;
        if (freecells[cell_mov_idx].item<int>() == 0) cout << "-------warn-------\n";
        // if (freecells[cell_mov_idx].item<int>() == 0) break;

        auto cell_mov = nodes[cell_mov_idx];

        // cout << gainlist.max().item<float>() / density_gainlist.max().item<float>() << endl;

        float gain = cost_list[cell_mov_idx].item<float>();
        if (macro_mask[cell_mov_idx].item<int>() == 1) {
            cout << gain << endl;
        }
        GAIN_ITER += gain;

        int cut_gain = cell_mov->gain;

        freecells[cell_mov_idx] = 0;
        num_free--;
        update_gain(cell_mov);
        update_gainWL(db, cell_mov_idx);

        cell_mov->group = !cell_mov->group;

        /* plt curve*/
        hpwls[i + 1] = hpwls[i] - gain;
        areas[i + 1] = (mov_cell_areas[0] / mov_cell_areas[1]).item<float>();
        cuts[i + 1] = cuts[i] - cut_gain;

        /* log swap cell to track */
        tracker.push_back(cell_mov_idx);
        cutsize -= cut_gain;
        hpwls_pass[i] = hpwls[i + 1];
        cuts_pass[i] = cutsize;
        gains_pass[i] = GAIN_ITER;
        areas_pass[i] = areas[i + 1];

        // if (GAIN_ITER > GAIN_MAX) {
        //     GAIN_MAX = GAIN_ITER;
        //     maxGAINIndex = i;
        // }
    }

    /* retrive max gain */
    cuts_pass /= num_nets;
    auto area_valid =
        torch::_cast_Float((areas_pass > upper_lower_bound_ratio[0]) * (areas_pass < upper_lower_bound_ratio[1]));

    auto gain_selector = area_valid * torch::pow((1 - cuts_pass), 1) * gains_pass;
    maxGAINIndex = torch::argmax(gain_selector, 0).item<int>();
    GAIN_MAX = gains_pass[maxGAINIndex].item<float>();

    // cout << "-----------------\n";
    // cout << maxGAINIndex << endl;
    // cout << hpwls_pass[maxGAINIndex] << endl;
    // cout << gain_selector[maxGAINIndex] << endl;
    // cout << cuts_pass[maxGAINIndex] << endl;
    // cout << gains_pass[maxGAINIndex] << endl;
    // cout << mov_cell_areas[0].item<float>() << " " << mov_cell_areas[1].item<float>() << endl;

    GAIN_FLOAT += GAIN_MAX;
    for (int i = maxGAINIndex + 1; i < tracker.size(); i++) {
        /* update cell info */
        int cell_mov_idx = tracker[i];
        swap_node(db, cell_mov_idx);
        nodes[cell_mov_idx]->group = db.node_die[cell_mov_idx];
    }

// #ifdef DEBUG
// #ifdef WITH_OPENCV
//     int n = num_swaps;
//     std::map<std::string, std::string> options = {
//         {"color", "r"},
//     };

//     auto res_root = logger.res_root;
//     std::filesystem::path eval_dir("eval");
//     std::filesystem::path fig_root = res_root / eval_dir;
//     if (!std::filesystem::exists(fig_root)) {
//         std::filesystem::create_directories(fig_root);
//     }
    
//     if (true) {
//         std::vector<double> x(n), y(n);
//         for (int i = 0; i < n; ++i) {
//             x.at(i) = i;
//             y.at(i) = hpwls[i];
//         }

//         plt::plot(x, y);
//         plt::axvline((float)maxGAINIndex, 0, 1, options);

//         std::string fig_path = fig_root.string() + "/[" + to_string(st::setting.round_recursion) + "]_fm_hpwl.png";
//         plt::save(fig_path);
//         plt::close();
//     }
//     plt::close();

//     if (true) {
//         std::vector<double> x(n), z(n);
//         for (int i = 0; i < n; ++i) {
//             x.at(i) = i;
//             z.at(i) = areas[i];
//         }

//         plt::plot(x, z);
//         plt::axvline((float)maxGAINIndex, 0, 1, options);
//         plt::axhline(upper_lower_bound_ratio[0].item<float>(), 0, 1, options);
//         plt::axhline(upper_lower_bound_ratio[1].item<float>(), 0, 1, options);

//         std::string fig_path = fig_root.string() + "/[" + to_string(st::setting.round_recursion) + "]_fm_area.png";
//         plt::save(fig_path);
//         plt::close();
//     }
//     plt::close();

//     if (true) {
//         std::vector<double> x(n), w(n);
//         for (int i = 0; i < n; ++i) {
//             x.at(i) = i;
//             w.at(i) = cuts[i];
//         }

//         plt::plot(x, w);
//         plt::axvline((float)maxGAINIndex, 0, 1, options);

//         std::string fig_path = fig_root.string() + "/[" + to_string(st::setting.round_recursion) + "]_fm_cuts.png";
//         plt::save(fig_path);
//         plt::close();
//     }
//     plt::close();
// #endif  // WITH_OPENCV
// #endif // DEBUG

}  // END MODULE

//-------------------------------------------------------------------------------

bool Partitioner::check_balance_global(int cell_mov_idx) {
    int group = node_die[cell_mov_idx].item<int>();
    if (((mov_cell_areas[!group] + node_areas[!group][cell_mov_idx]) <
         st::setting.soft_margin * max_mov_cell_areas[!group])
            .item<bool>()) {
        return true;
    } else
        return false;
}  // END MODULE

//-------------------------------------------------------------------------------

int Partitioner::pop_maxWL() {
    auto gainlist_0 = gainlist - node_die * max_hpwl;
    int cell_mov_idx_0 = torch::argmax(gainlist_0, 0).item<int>();
    auto gainlist_1 = gainlist - (1 - node_die) * max_hpwl;
    int cell_mov_idx_1 = torch::argmax(gainlist_1, 0).item<int>();
    bool check0 = check_balance_global(cell_mov_idx_0);
    bool check1 = check_balance_global(cell_mov_idx_1);

    if (check0 && check1) {
        return ((gainlist[cell_mov_idx_0] > gainlist[cell_mov_idx_1]).item<bool>() ? cell_mov_idx_0 : cell_mov_idx_1);
    } else if (check0)
        return cell_mov_idx_0;
    else if (check1)
        return cell_mov_idx_1;
    else {
        return -1;
    }
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::swap_node(pt::PartitionData& db, int cell_mov) {
    int c_id = db.node_die[cell_mov];
    swap_c_id = 1 - c_id;
    int other_c_id = 1 - c_id;
    db.node_die[cell_mov] = other_c_id;
    gainlist[cell_mov] = -std::numeric_limits<float>::max();
    vector<int>().swap(surround_bin_id);
    for (int node2pin_id = db.flat_node2pin_start_map[cell_mov]; node2pin_id < db.flat_node2pin_start_map[cell_mov + 1];
         ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        db.pin_offset_x[node_pin_id] = db.pin_offset_xs[other_c_id][node_pin_id];
        db.pin_offset_y[node_pin_id] = db.pin_offset_ys[other_c_id][node_pin_id];
    }

    cell_xl = torch::zeros({2}, dtype(torch::kInt));
    cell_xh = torch::zeros({2}, dtype(torch::kInt));
    cell_yl = torch::zeros({2}, dtype(torch::kInt));
    cell_yh = torch::zeros({2}, dtype(torch::kInt));
    for (int n = 0; n < 2; n++) {
        int c_id_ = (n == 0) ? c_id : (1 - c_id);
        float node_xl = mov_node_xl[c_id_][cell_mov].item<float>();
        float node_xh = mov_node_xh[c_id_][cell_mov].item<float>();
        float node_yl = mov_node_yl[c_id_][cell_mov].item<float>();
        float node_yh = mov_node_yh[c_id_][cell_mov].item<float>();
        int node_xl_b = mov_node_xl_b[c_id_][cell_mov].item<int>();
        int node_yl_b = mov_node_yl_b[c_id_][cell_mov].item<int>();
        int node_xh_b = mov_node_xh_b[c_id_][cell_mov].item<int>();
        int node_yh_b = mov_node_yh_b[c_id_][cell_mov].item<int>();
        cell_xl[c_id_] = node_xl_b;
        cell_yl[c_id_] = node_yl_b;
        cell_xh[c_id_] = node_xh_b;
        cell_yh[c_id_] = node_yh_b;

        // cout << "swap node: " << node_xl_b << " " << node_xh_b << " " << node_yl_b << " " << node_yh_b << endl;

        for (int j = node_xl_b; j < node_xh_b + 1; j++) {
            float bin_x_l = static_cast<float>(j);
            float overlap_x = overlap(node_xl, node_xh, bin_x_l);
            for (int k = node_yl_b; k < node_yh_b + 1; k++) {
                float bin_y_l = static_cast<float>(k);
                float overlap_y = overlap(node_yl, node_yh, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                density_map[c_id_][j][k] -= (n == 0) ? overlap_area : (-overlap_area);
                if (surround_bin_id.size() == 0 ||
                    std::find(surround_bin_id.begin(), surround_bin_id.end(), j * num_y_bin + k) ==
                        surround_bin_id.end()) {
                    surround_bin_id.emplace_back(j * num_y_bin + k);
                }
            }
        }
    }
    density_gainlist[cell_mov] = -std::numeric_limits<float>::max();
    mov_cell_areas[c_id] -= node_areas[c_id][cell_mov];
    mov_cell_areas[other_c_id] += node_areas[other_c_id][cell_mov];
    // mov_cell_areas[c_id] -= nodes[cell_mov]->sizes[c_id];
    // mov_cell_areas[other_c_id] += nodes[cell_mov]->sizes[other_c_id];
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::update_gainWL(pt::PartitionData& db, int cell_mov) {
    swap_node(db, cell_mov);
    for (shared_ptr<ptNet> net : nodes[cell_mov]->Nets) {
        int i = net->id;
        if (!db.net_mask[i]) continue;
        vector<Box> boxs;
        boxs.emplace_back(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max());
        boxs.emplace_back(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max());

        vector<Box> box2s;
        box2s.emplace_back(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max());
        box2s.emplace_back(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max());

        vector<int> num_node_count(2, 0);
        for (int net2pin_id = db.flat_net2pin_start_map[i]; net2pin_id < db.flat_net2pin_start_map[i + 1];
             ++net2pin_id) {
            int net_pin_id = db.flat_net2pin_map[net2pin_id];
            int node_id = db.pin2node_map[net_pin_id];
            int c_id = db.node_die[node_id];
            num_node_count[c_id]++;

            float xx = db.x[node_id] + db.pin_offset_xs[c_id][net_pin_id];
            float yy = db.y[node_id] + db.pin_offset_ys[c_id][net_pin_id];

            if (xx > boxs[c_id].xh) {
                box2s[c_id].xh = boxs[c_id].xh;
                boxs[c_id].xh = xx;
            } else if (xx <= boxs[c_id].xh && xx > box2s[c_id].xh)
                box2s[c_id].xh = xx;
            if (yy > boxs[c_id].yh) {
                box2s[c_id].yh = boxs[c_id].yh;
                boxs[c_id].yh = yy;
            } else if (yy <= boxs[c_id].yh && yy > box2s[c_id].yh)
                box2s[c_id].yh = yy;

            if (xx < boxs[c_id].xl) {
                box2s[c_id].xl = boxs[c_id].xl;
                boxs[c_id].xl = xx;
            } else if (xx >= boxs[c_id].xl && xx < box2s[c_id].xl)
                box2s[c_id].xl = xx;
            if (yy < boxs[c_id].yl) {
                box2s[c_id].yl = boxs[c_id].yl;
                boxs[c_id].yl = yy;
            } else if (yy >= boxs[c_id].yl && yy < box2s[c_id].yl)
                box2s[c_id].yl = yy;
        }

        for (int net2pin_id = db.flat_net2pin_start_map[i]; net2pin_id < db.flat_net2pin_start_map[i + 1];
             ++net2pin_id) {
            int net_pin_id = db.flat_net2pin_map[net2pin_id];
            int node_id = db.pin2node_map[net_pin_id];
            if (freecells[node_id].item<int>() == 0) continue;  // FIXME:
            int c_id = db.node_die[node_id];
            int other_c_id = 1 - c_id;
            if (num_node_count[c_id] < 2) {
                // clear this swap
                if (macro_mask[node_id].item<int>() != 1) {
                    gainlist[node_id] += -nodes[node_id]->gain_map[i];
                    nodes[node_id]->gain_map[i] = 0;
                }
                continue;
            }

            vector<Box> box_swaps;
            box_swaps.emplace_back(boxs[0]);
            box_swaps.emplace_back(boxs[1]);

            float xx = db.x[node_id] + db.pin_offset_xs[c_id][net_pin_id];
            if (xx == boxs[c_id].xh) box_swaps[c_id].xh = box2s[c_id].xh;
            if (xx == boxs[c_id].xl) box_swaps[c_id].xl = box2s[c_id].xl;
            float other_xx = db.x[node_id] + db.pin_offset_xs[other_c_id][net_pin_id];
            if (other_xx > boxs[other_c_id].xh) box_swaps[other_c_id].xh = other_xx;
            if (other_xx < boxs[other_c_id].xl) box_swaps[other_c_id].xl = other_xx;

            float yy = db.y[node_id] + db.pin_offset_ys[c_id][net_pin_id];
            if (yy == boxs[c_id].yh) box_swaps[c_id].yh = box2s[c_id].yh;
            if (yy == boxs[c_id].yl) box_swaps[c_id].yl = box2s[c_id].yl;
            float other_yy = db.y[node_id] + db.pin_offset_ys[other_c_id][net_pin_id];
            if (other_yy > boxs[other_c_id].yh) box_swaps[other_c_id].yh = other_yy;
            if (other_yy < boxs[other_c_id].yl) box_swaps[other_c_id].yl = other_yy;

            // float x_max_mid = min(boxs[0].xh, boxs[1].xh);
            // float x_min_mid = max(boxs[0].xl, boxs[0].xl);
            // float x_ovlp = max(x_max_mid - x_min_mid, (float)0);
            // float x_max_mid_swap = min(box_swaps[0].xh, box_swaps[1].xh);
            // float x_min_mid_swap = max(box_swaps[0].xl, box_swaps[0].xl);
            // float x_ovlp_swap = max(x_max_mid_swap - x_min_mid_swap, (float)0);
            // float gain_x = x_ovlp - x_ovlp_swap;

            // float y_max_mid = min(boxs[0].yh, boxs[1].yh);
            // float y_min_mid = max(boxs[0].yl, boxs[0].yl);
            // float y_ovlp = max(y_max_mid - y_min_mid, (float)0);
            // float y_max_mid_swap = min(box_swaps[0].yh, box_swaps[1].yh);
            // float y_min_mid_swap = max(box_swaps[0].yl, box_swaps[0].yl);
            // float y_ovlp_swap = max(y_max_mid_swap - y_min_mid_swap, (float)0);
            // float gain_y = y_ovlp - y_ovlp_swap;

            // gain = gain_x + gain_y;
            // gainlist[node_id] += gain - nodes[node_id]->gain_map[i];
            // nodes[node_id]->gain_map[i] = gain;

            float wl = boxs[0] + boxs[1];
            float wl_swap = box_swaps[0] + box_swaps[1];
            float gain = wl - wl_swap;
            
            if (macro_mask[node_id].item<int>() != 1) {
                gainlist[node_id] += gain - nodes[node_id]->gain_map[i];
                nodes[node_id]->gain_map[i] = gain;
            }
        }
    }
    
    auto density_map_a = density_map.accessor<float, 3>();
    auto mov_node_xl_a = mov_node_xl.accessor<float, 2>();
    auto mov_node_xh_a = mov_node_xh.accessor<float, 2>();
    auto mov_node_yl_a = mov_node_yl.accessor<float, 2>();
    auto mov_node_yh_a = mov_node_yh.accessor<float, 2>();
    auto mov_node_xl_b_a = mov_node_xl_b.accessor<int, 2>();
    auto mov_node_xh_b_a = mov_node_xh_b.accessor<int, 2>();
    auto mov_node_yl_b_a = mov_node_yl_b.accessor<int, 2>();
    auto mov_node_yh_b_a = mov_node_yh_b.accessor<int, 2>();
    auto cell_xl_a = cell_xl.accessor<int, 1>();
    auto cell_xh_a = cell_xh.accessor<int, 1>();
    auto cell_yl_a = cell_yl.accessor<int, 1>();
    auto cell_yh_a = cell_yh.accessor<int, 1>();

    auto has_not_updated = torch::ones(num_nodes, dtype(torch::kInt));
    for (auto i : surround_bin_id) {
        for (auto m : bin2node_id[i]) {
            if (has_not_updated[m].item<int>() == 1) {
                int node_id = m;
                has_not_updated[node_id] = 0;
                if (freecells[node_id].item<int>() == 0) continue;
                
                int c_id = db.node_die[node_id];
                float gain_density = 0;
                for (int n = 0; n < 2; n++) {
                    int c_id_ = (n == 0) ? c_id : (1 - c_id);
                    float node_xl = mov_node_xl_a[c_id_][node_id];
                    float node_xh = mov_node_xh_a[c_id_][node_id];
                    float node_yl = mov_node_yl_a[c_id_][node_id];
                    float node_yh = mov_node_yh_a[c_id_][node_id];
                    int node_xl_b = mov_node_xl_b_a[c_id_][node_id];
                    int node_yl_b = mov_node_yl_b_a[c_id_][node_id];
                    int node_xh_b = mov_node_xh_b_a[c_id_][node_id];
                    int node_yh_b = mov_node_yh_b_a[c_id_][node_id];
                    float current_density = 0;

                    // cout << "surround node: " << node_xl_b << " " << node_xh_b << " " << node_yl_b << " " << node_yh_b << endl;

                    for (int j = node_xl_b; j < node_xh_b + 1; j++) {
                        float bin_x_l = static_cast<float>(j);
                        float overlap_x = overlap(node_xl, node_xh, bin_x_l);
                        for (int k = node_yl_b; k < node_yh_b + 1; k++) {
                            float bin_y_l = static_cast<float>(k);
                            float overlap_y = overlap(node_yl, node_yh, bin_y_l);
                            float overlap_area = overlap_x * overlap_y;
                            current_density += overlap_area * (density_map_a[c_id_][j][k] + ((n == 0) ? 0 : overlap_area));
                        }
                    }
                    gain_density += (n == 0) ? current_density : (-current_density);
                }
                density_gainlist[node_id] = gain_density;
            }
        }
    }

}  // END MODULE