#include "../run_placement.h"
vector<int> mark_is_cut(torch::Tensor node_die, NodeData& data) {
    torch::Tensor net_cut_info = torch::zeros({data.num_nets, 2}, torch::dtype(torch::kInt));
    auto pin_id2node_id_cpu = data.pin_id2node_id.clone().cpu();
    const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = pin_id2node_id_cpu.accessor<int64_t, 1>();
    auto hyperedge_list_cpu = data.hyperedge_list.clone().cpu();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = hyperedge_list_cpu.accessor<int64_t, 1>();
    auto hyperedge_list_end_cpu = data.hyperedge_list_end.clone().cpu();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = hyperedge_list_end_cpu.accessor<int64_t, 1>();
    auto node_die_cpu = node_die.clone().cpu();
    const torch::TensorAccessor<int, 1> cell_die_at = node_die_cpu.accessor<int, 1>();
    // int lx0=2000000, ly0=2000000, rx0=-20000000, ry0=-20000000;
    // int lx1=2000000, ly1=2000000, rx1=-20000000, ry1=-20000000;
    // vector<int> regionlx;
    // regionlx.resize(data.cell_mov_rhs)
    // vector<int> regionly;
    // regionly.resize(data.cell_mov_rhs)
    // vector<int> regionrx;
    // regionrx.resize(data.cell_mov_rhs)
    // vector<int> regionry;
    // regionry.resize(data.cell_mov_rhs)
    for (int i = 0; i != data.num_nets; i++) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end_at[i - 1];
        }
        int64_t end_idx = hyperedge_list_end_at[i];
        if (end_idx != start_idx) {
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list_at[idx];
                int64_t node_id = pin_id2node_id_at[pin_id];
                if(node_id>data.cell_mov_rhs)
                {
                    continue;
                }
                if (node_id > data.num_nodes) continue;

                int c_id = cell_die_at[node_id];

                if (c_id != 0 && c_id != 1) continue;

                net_cut_info[i][c_id] += 1;
            }
        }
    }
    vector<int> mark_is_cut_vector;
    mark_is_cut_vector.resize(data.num_nets);
    for (int i = 0; i != data.num_nets; i++) {
        if ((torch::prod(net_cut_info[i], 0) != 0).item<int>()) {
            mark_is_cut_vector[i] = 1;
        }
    }
    return mark_is_cut_vector;
}  // END MODULE
torch::Tensor run_gp(NodeData& data,
                     ViaData& via_data,
                     torch::Tensor node_pos,
                     states& hpwl_state,
                     int& cell_mov_lhs,
                     int& cell_mov_rhs,
                     int& via_mov_lhs,
                     int& via_mov_rhs,
                     int& mov_lhs,
                     int& mov_rhs,
                     bool move_macro,
                     bool is_init_macro,
                     bool is_init_stdcell,
                     string message) {
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};
    auto node_die = data.node_die.clone();
    auto device = data.device;

    st::setting.use_filler = true;
    st::setting.early_stop_check_plateau = false;
    // st::setting.magic_hpwl *= 2;
    st::setting.density_weight_coef -= 0.01;
    st::setting.wa_coeff /= 2;
    st::setting.target_density = 1;
    // st::setting.stop_overflow = 0.3;
    data.target_density = st::setting.target_density;

    logger.info("Optimizer info: density_weight: %3E | wa_coef %.2f | magic_hpwl %d",
                st::setting.density_weight,
                st::setting.wa_coeff,
                st::setting.magic_hpwl);

    /* cell visualization information */

    torch::Tensor node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
    torch::Tensor node_size_top = data.node_size_top * node_die.unsqueeze(1);

    int checkk = node_size_bot[0][1].item<int>();
    checkk = node_size_top[0][1].item<int>();

    /* movable cells */
    // torch::Tensor init_density_map = get_init_density_map(data);
    torch::Tensor init_density_maps = get_init_density_map_cross_chip(data);
    data.init_density_maps = data.init_density_maps.cpu();  // TODO:
    // bool is_init_macro = true;
    // bool is_init_stdcell = true;
    if (st::setting.use_floorplan) {
        is_init_macro = false;
        is_init_stdcell = true;
    }
    auto [mov_node_pos, mov_node_size, expand_ratio] =
        data.get_mov_node_info_cross_chip(is_init_macro, is_init_stdcell, move_macro);
    std::tie(mov_lhs, mov_rhs) = data.movable_index;

    torch::Tensor via_init_density_map =
        torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);
    if (st::setting.block_row) {
        torch::Tensor zeros_density_map =
            torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);

        torch::Tensor node_pos =
            torch::zeros({via_data.numRows.item<int>(), 2}, torch::dtype(torch::kFloat).device(device));
        torch::Tensor node_size =
            torch::zeros({via_data.numRows.item<int>(), 2}, torch::dtype(torch::kFloat).device(device));
        torch::Tensor node_weight =
            torch::ones(via_data.numRows.item<int>(), torch::dtype(torch::kFloat).device(device));

        node_size.index_put_({"...", 0}, via_data.core_info[1]);
        node_size.index_put_({"...", 1}, via_data.row_height * 0.1);

        node_pos.index_put_({"...", 0}, (via_data.core_info[0] + via_data.core_info[1]) * 0.5);
        for (int r = 0; r < via_data.numRows.item<int>(); r++) {
            node_pos[r][1] = via_data.row_height * (r + 1);
        }
        /*
        via_init_density_map = density_map_forward_naive(
            node_pos.to(device), node_size.to(device), node_weight.to(device), data.unit_len.to(device),
            zeros_density_map.to(device), data.num_bin_x, data.num_bin_y, node_pos.sizes()[0], -1.0, -1.0, 1e-4, false);
        */
        via_init_density_map = density_map_forward_naive(node_pos.to(device),
                                                         node_size.to(device),
                                                         node_weight.to(device),
                                                         data.unit_len.to(device),
                                                         zeros_density_map.to(device),
                                                         data.num_bin_x,
                                                         data.num_bin_y,
                                                         node_pos.sizes()[0],
                                                         -1.0,
                                                         -1.0,
                                                         1e-4,
                                                         false,
                                                         true);
    }
    init_density_maps = torch::cat({init_density_maps, via_init_density_map.unsqueeze(0)}, 0);

    via_data.init_vars();  // TODO: equivalent to PlaceData::init_filler
    auto [via_mov_node_pos, via_mov_node_size, via_expand_ratio] = via_data.get_mov_node_info();
    std::tie(via_mov_lhs, via_mov_rhs) = via_data.movable_index;

    torch::Tensor mov_node_pos_all;
    torch::Tensor mov_node_size_all;
    torch::Tensor expand_ratio_all;
    /* movable cells&vias*/
    std::tie(mov_node_pos_all, mov_node_size_all, expand_ratio_all) =
        data.get_mov_node_info_cross_chip_with_via(via_data,
                                                   make_tuple(mov_node_pos, mov_node_size, expand_ratio),
                                                   make_tuple(via_mov_node_pos, via_mov_node_size, via_expand_ratio));
    std::tie(mov_lhs, mov_rhs) = data.movable_index;
    if (node_pos.numel() && st::setting.use_pre_gp) {  // if use_pre_gp = true, it will not initialize during gp step
        int cell_mov_rhs_min = std::min(mov_node_pos_all.size(0), node_pos.size(0));
        logger.info("load from prev sol of size %d...", cell_mov_rhs_min);

        auto mov_node_pos_all_copy = mov_node_pos_all.clone();
        mov_node_pos_all_copy.index({Slice(0, cell_mov_rhs_min), Slice(0, 2)})
            .data()
            .copy_(node_pos.index({Slice(0, cell_mov_rhs_min), Slice(0, 2)}).data());

        if (st::setting.use_pre_gp) mov_node_pos_all = mov_node_pos_all_copy;
    }
    mov_node_pos_all = mov_node_pos_all.to(data.device).requires_grad_(true);
    mov_node_size_all = mov_node_size_all.to(data.device);
    auto mov_node_size_all_backup = mov_node_size_all.clone();
    checkk = mov_node_size_all[0][1].item<int>();
    expand_ratio_all = expand_ratio_all.to(data.device);
    data.to(data.device);

    /* trunc nodes to core */
    torch::Tensor node_pos_lb = mov_node_size_all / 2 + data.mov_node_sideline_ll + 1e-4;
    torch::Tensor node_pos_ub = data.mov_node_sideline_ur - mov_node_size_all / 2 + data.die_ll - 1e-4;
    // if(!move_macro)
    // if(true)
    // {
    //     auto single_sideline_ll = data.die_ll - 1e-4;
    //     auto single_sideline_ur = data.die_ur + data.die_ll - 1e-4;

    //     auto node_pos_lb_macro = single_sideline_ll.repeat({data.num_nodes, 1}) + mov_node_size_all.slice(0,0,data.num_nodes) / 2;
    //     auto node_pos_ub_macro = single_sideline_ll.repeat({data.num_nodes, 1}) - mov_node_size_all.slice(0,0,data.num_nodes) / 2;

    //     auto node_pos_lb_std = mov_node_size_all / 2 + data.mov_node_sideline_ll + 1e-4;
    //     auto node_pos_ub_std = data.mov_node_sideline_ur - mov_node_size_all / 2 + data.die_ll - 1e-4;
        
    //     node_pos_lb.slice(0,0,data.num_nodes) = node_pos_lb_std.slice(0,0,data.num_nodes)*(1-data.macro_mask.unsqueeze(1)).slice(0,0,data.num_nodes)
    //                       +node_pos_lb_macro.slice(0,0,data.num_nodes)*data.macro_mask.unsqueeze(1).slice(0,0,data.num_nodes);
    //     node_pos_ub.slice(0,0,data.num_nodes) = node_pos_ub_std.slice(0,0,data.num_nodes)*(1-data.macro_mask.unsqueeze(1)).slice(0,0,data.num_nodes)
    //                       +node_pos_ub_macro.slice(0,0,data.num_nodes)*data.macro_mask.unsqueeze(1).slice(0,0,data.num_nodes);
    // }
    // for(auto macro_id:data.macro_list)
    // {
    //     node_pos_lb[macro_id]-=st::setting.sideline;
    //     node_pos_ub[macro_id]+=st::setting.sideline;
    // }
    std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn = [&node_pos_lb, &node_pos_ub](torch::Tensor x) {
        x.data().clamp_(node_pos_lb, node_pos_ub);
        return x;
    };

    /* parameteer scheduler */
    ParamScheduler ps = ParamScheduler(data);

    /* overflow function for each electrostatic field */
    vector<ElectronicDensityLayer> density_map_layers;
    torch::Tensor mov_cell_areas_without_macro = data.mov_cell_areas.clone();
    if (!move_macro) {
        auto size_bot = data.node_size_bot.clone().to(data.device).slice(0, 0, data.macro_mask.size(0));
        auto size_top = data.node_size_top.clone().to(data.device).slice(0, 0, data.macro_mask.size(0));
        mov_cell_areas_without_macro[0] -= torch::sum(
            ((1 - data.node_die.slice(0, 0, data.macro_mask.size(0))) * torch::prod(size_bot, 1)) * data.macro_mask);
        mov_cell_areas_without_macro[1] -= torch::sum(
            ((data.node_die.slice(0, 0, data.macro_mask.size(0))) * torch::prod(size_top, 1)) * data.macro_mask);
    }

    for (int i = 0; i < 3; i++) {
        std::function<torch::Tensor(torch::Tensor)> overflow_fn_cc =
            [&data, i, &mov_cell_areas_without_macro](torch::Tensor mov_density_map) {
                torch::Tensor overflow_sum = ((mov_density_map - st::setting.target_density) * data.bin_area)
                                                 .clamp_(0.0)
                                                 .sum();  // TODO: each layer different ovfl
                // torch::Tensor area_sum = (mov_density_map * data.bin_area)
                //                                  .clamp_(0.0)
                //                                  .sum();  // TODO: each layer different ovfl
                // return overflow_sum / area_sum;
                return overflow_sum / mov_cell_areas_without_macro[i];
            };
        auto overflow_helper_cc = make_tuple(mov_lhs, mov_rhs, overflow_fn_cc);
        density_map_layers.emplace_back(data.unit_len,
                                        data.num_bin_x,
                                        data.num_bin_y,
                                        device,
                                        overflow_helper_cc,
                                        expand_ratio_all,
                                        data.sorted_maps,
                                        data.macro_mask);
    }
    // auto tmp = density_map_layers[2].forward(mov_node_pos_all, mov_node_size_all, init_density_map,
    // data.mov_node_weights[0]); cout << density_map_layers[2].cache_node_weight.sum() << endl;
    auto tmp = density_map_layers[2].forward(
        mov_node_pos_all, mov_node_size_all, init_density_maps[2], data.mov_node_weights[2]);

    /* objective function */
    /* follow the order of node types */
    /* | MovConnected | MovFloat | ConnFixed ... | */
    torch::Tensor conn_fix_node_pos = data.node_pos.new_empty({0, 2});
    if (get<0>(data.fixed_connected_index) < get<1>(data.fixed_connected_index)) {
        auto [lhs, rhs] = data.fixed_connected_index;
        conn_fix_node_pos = data.node_pos.index({Slice(lhs, rhs), "..."});
    }
    conn_fix_node_pos = conn_fix_node_pos.detach();
    torch::Tensor macro_mask = data.macro_mask;
    macro_mask = macro_mask.to(device);

    vector<int> macro_list;
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        if (data.macro_mask[i].item<int>() == 1) {
            macro_list.push_back(i);
        }
    }
    int iter_num = 0;
    double speed = 2;
    double multiplier = 0.999;
    torch::Tensor grad_tmp = torch::zeros({mov_node_size.size(0), 2});
    std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> obj_and_grad_fn =
        [&trunc_node_pos_fn,
         &mov_node_size_all,
         &init_density_maps,
         &density_map_layers,
         &conn_fix_node_pos,
         &ps,
         &data,
         &via_data,
         &move_macro,
         &macro_list,
         &iter_num,
         &speed,
         &multiplier,
         &grad_tmp,
         &mov_node_size_all_backup](at::Tensor mov_node_pos_all) {
            auto [loss, grad] = calc_obj_and_grad_multi_circuit(mov_node_pos_all,
                                                                trunc_node_pos_fn,
                                                                mov_node_size_all,
                                                                init_density_maps,
                                                                density_map_layers,
                                                                conn_fix_node_pos,
                                                                ps,
                                                                data,
                                                                via_data);
            iter_num++;
            if (!move_macro) {
                for (auto macro_id : macro_list) {
                    grad[macro_id] = 0;
                }
            }
            if (st::setting.skip_2d) {
                grad.index({torch::indexing::Slice(data.cell_mov_lhs, data.cell_mov_rhs), torch::indexing::Slice(0, 2)}) = 0.0;
            } else {
                grad.index({torch::indexing::Slice(data.iopin_mov_lhs, data.iopin_mov_rhs), torch::indexing::Slice(0, 2)}) = 0.0;
            }
            return std::make_tuple(loss, grad);
        };

    /* evaluation function */
    std::function<tuple<torch::Tensor, torch::Tensor, torch::Tensor>(torch::Tensor)> evaluator_fn =
        [&trunc_node_pos_fn,
         &mov_node_size_all,
         &init_density_maps,
         &density_map_layers,
         &conn_fix_node_pos,
         &ps,
         &data](at::Tensor mov_node_pos_all) {
            return fast_evaluator_multi_circuit(mov_node_pos_all,
                                                trunc_node_pos_fn,
                                                mov_node_size_all,
                                                init_density_maps,
                                                density_map_layers,
                                                conn_fix_node_pos,
                                                ps,
                                                data);
        };

    /* Nesterov optimizer */
    auto optimizer = torch::optim::Nesterov({mov_node_pos_all}, torch::optim::NesterovOptions(0.0), obj_and_grad_fn);

    init_params_multi_circuit(mov_node_pos_all,
                              trunc_node_pos_fn,
                              mov_lhs,
                              mov_rhs,
                              conn_fix_node_pos,
                              density_map_layers,
                              mov_node_size_all,
                              init_density_maps,
                              optimizer,
                              ps,
                              data);

    /* learning rate */
    double init_lr =
        estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos_all, st::setting.lr);
    logger.info("Init learning rate %.3E", init_lr);
    for (auto& group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::NesterovOptions&>(group.options());
        options.set_lr(init_lr);
    }

    /* start gp iteration */
    if (true) {
        auto [hpwls, overflows, tmp1] = evaluator_fn(mov_node_pos_all);
        logger.info(
            "iter: %d | masked_hpwl: %.2E overflow: (%.4f, %.4f, %.4f) "
            "density_weight: %.4E wa_coeff: %.4E",
            0,
            hpwls.sum().item<float>(),
            overflows[0].item<float>(),
            overflows[1].item<float>(),
            overflows[2].item<float>(),
            ps.density_weight,
            ps.wa_coeff);

        auto mov_node_size_all_draw = mov_node_size_all.to(torch::kCPU).clone();
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(mov_node_pos_all).to(torch::kCPU);
        auto true_mov_node_pos_all = mov_node_pos_all.to(torch::kCPU) + node_shift;
        // mov_node_size_all_draw.index({Slice(mov_lhs, mov_rhs)}) *= 0;
        auto info3 = make_tuple(st::setting.round_recursion, 0, message + data.design_name + "_PT_GP_INIT_2");
        draw_fig_with_cairo_cpp(true_mov_node_pos_all.to(torch::kCPU), mov_node_size_all_draw, data, info3);
    }
    // data.init_shape_params(mov_node_size_all);
    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    int& iteration = st::setting.iteration;
    // if (st::setting.use_pre_gp) {
    //     logger.info("=========================================");
    //     logger.info("start pre_gp, use wl loss");
    //     st::setting.loss_type = "wl_only";
    //     int v = 50;
    //     for (iteration = 0; iteration < v; iteration++) {
    //         torch::Tensor obj = optimizer.step();
    //         auto [hpwls, overflows, tmp1] = evaluator_fn(mov_node_pos_all);

    //         if (iteration == v - 1) {
    //             logger.info(
    //                 "iter: %-5d | masked_hpwl: %.2E overflow: (%.4f, %.4f, %.4f) obj: %.4E "
    //                 "density_weight: %.4E wa_coeff: %.4E",
    //                 iteration,
    //                 hpwls.sum().item<float>(),
    //                 overflows[0].item<float>(),
    //                 overflows[1].item<float>(),
    //                 overflows[2].item<float>(),
    //                 obj.item<float>(),
    //                 ps.density_weight,
    //                 ps.wa_coeff);
    //         }
    //     }
    //     st::setting.loss_type = "direct";
    // }
        ////////////////////////////////////////////////////////////////////////////////
    vector<int> ocupied;
    vector<int> mark_is_cut_vector = mark_is_cut(data.node_die, data);
    ocupied.resize(data.num_pins);
    int cnt_violate = 0;
    float total_region_x = 0;
    float total_region_y = 0;
    for (int i = 0; i < data.num_nets; i++) {
        if (mark_is_cut_vector[i] == 0) {
            continue;
        }
        int net_id = i;
        int start_idx = 0;
        if (net_id > 0) {
            start_idx = data.hyperedge_list_end[net_id - 1].item<int>();
        }
        int end_idx = data.hyperedge_list_end[net_id].item<int>();
        int choose = -1;
        int min_occupied = 2000000;
        int min_occupied_pin_id = -1;
        float lx0 = -2000000, ly0 = -2000000, rx0 = 20000000, ry0 = 20000000;
        float lx1 = -2000000, ly1 = -2000000, rx1 = 20000000, ry1 = 20000000;
        for (int ii = start_idx; ii < end_idx; ii++) {
            int pin_id = data.hyperedge_list[ii].item<int>();
            // if (ocupied[pin_id] == 0) {
            //     choose = pin_id;
            //     break;
            // } else {
            //     if(ocupied[pin_id]<min_occupied)
            //     {
            //         min_occupied = ocupied[pin_id];
            //         min_occupied_pin_id = pin_id;
            //     }
            // }
            int node_id = data.pin_id2node_id[pin_id].item<int>();
            if (node_id >= data.cell_mov_rhs) {
                continue;
            }
            float node_pos_x = node_pos[node_id][0].item<float>();
            float node_pos_y = node_pos[node_id][1].item<float>();
            float pin_rel_cpos_x = data.pin_rel_cpos[pin_id][0].item<float>();
            float pin_rel_cpos_y = data.pin_rel_cpos[pin_id][1].item<float>();
            float pin_x = node_pos_x + pin_rel_cpos_x;
            float pin_y = node_pos_y + pin_rel_cpos_y;
            int die_id = node_die[node_id].item<int>();
            if (die_id == 0) {
                lx0 = max(lx0, pin_x);
                ly0 = max(ly0, pin_y);
                rx0 = min(rx0, pin_x);
                ry0 = min(ry0, pin_y);
            } else {
                lx1 = max(lx1, pin_x);
                ly1 = max(ly1, pin_y);
                rx1 = min(rx1, pin_x);
                ry1 = min(ry1, pin_y);
            }
        }
        float llx = max(rx0, rx1);
        float rrx = min(lx0, lx1);
        float lly = max(ry0, ry1);
        float rry = min(ly0, ly1);

        float region_lx = min(llx, rrx);
        float region_rx = max(llx, rrx);
        float region_ly = min(lly, rry);
        float region_ry = max(lly, rry);
        if (abs(llx) > 100000 || abs(rrx) > 100000 || abs(lly) > 100000 || abs(rry) > 100000) {
            int debuggg = 0;
        }

        float region_center_x = (region_lx + region_rx) / 2;
        float region_center_y = (region_ly + region_ry) / 2;

        total_region_x += region_rx - region_lx;
        total_region_y += region_ry - region_ly;

        if (choose == -1) {
            cnt_violate++;
            choose = min_occupied_pin_id;
        }
        // ocupied[choose]++;
        // int node_id = data.pin_id2node_id[choose].item<int>();
        // float node_pos_x = node_pos[node_id][0].item<float>();
        // float node_pos_y = node_pos[node_id][1].item<float>();
        // float pin_rel_cpos_x = data.pin_rel_cpos[choose][0].item<float>();
        // float pin_rel_cpos_y = data.pin_rel_cpos[choose][1].item<float>();l
        // float pin_x = node_pos_x + pin_rel_cpos_x;
        // float pin_y = node_pos_y + pin_rel_cpos_y;
        torch::NoGradGuard no_grad;
        // mov_node_pos_all[data.cell_mov_rhs+i][0] = pin_x;
        // mov_node_pos_all[data.cell_mov_rhs+i][1] = pin_y;
        mov_node_pos_all[data.cell_mov_rhs + i][0] = region_center_x;
        mov_node_pos_all[data.cell_mov_rhs + i][1] = region_center_y;
    }
    total_region_x /= data.num_nets;
    total_region_y /= data.num_nets;
    logger.info("cnt_violate: %d", cnt_violate);
    logger.info("average region width: %f, average region height: %f", total_region_x, total_region_y);
    ////////////////////////////////////////////////////////////////////////////////
    logger.info("=========================================");
    logger.info("start gp");
    iteration = 0;  // FIXME: 0 ? 1
    if (true) {
        auto [hpwls, overflows, tmp1] = evaluator_fn(mov_node_pos_all);
        logger.info(
            "Optimal Place | masked_hpwl: %.2E overflow: (%.4f, %.4f, %.4f) "
            "density_weight: %.4E wa_coeff: %.4E",
            hpwls.sum().item<float>(),
            overflows[0].item<float>(),
            overflows[1].item<float>(),
            overflows[2].item<float>(),
            ps.density_weight,
            ps.wa_coeff);

        ps.cur_overflows = overflows;
        ps.step_wa_coeffs();
    }
    int iter_time = st::setting.inner_iter;
    // auto [hpwl_now, overflows_now, tmp_now] = evaluator_fn(mov_node_pos_all);
    // if (overflows_now[2].item<float>() < st::setting.stop_overflow_via) {
    //     iter_time = 0;
    //     logger.info("overflows are good enough, no need gp");
    // }
    
    for (iteration = 0; iteration < iter_time; iteration++) {
        // for (iteration = 0; iteration < 2000; iteration++) {
        torch::Tensor obj = optimizer.step();
        // data.updata_shape_by_density_grad(st::setting.cache_density_grad_4part, mov_node_size_all, iteration);
        // data.update_macro_orientaion_by_pin_std(mov_node_pos);
        // data.update_macro_orientaion_by_pin_std(mov_node_pos_all);
        auto [hpwls, overflows, tmp1] = evaluator_fn(mov_node_pos_all);
        // ps.step(hpwls.sum().item<float>(), overflows.index({Slice(0, 2)}).sum().item<float>() / 2, mov_node_pos_all);
        ps.steps(hpwls, overflows, mov_node_pos_all);
        // if (st::setting.num_den_layer == 3 && !st::setting.skip_2d) {
        //     if (overflows[2].item<float>() < st::setting.stop_overflow_via) {
        //         st::setting.num_den_layer = 2;
        //     }
        // }
        if (iteration % st::setting.log_freq == 0 || iteration == st::setting.inner_iter - 1 ||
            (iteration >= st::setting.minGPStep && ps.need_to_early_stop())) {
            logger.info(
                "iter: %-5d | masked_hpwl: %.2E overflow: (%.4f, %.4f, %.4f) obj: %.4E "
                "density_weight: %.4E wa_coeff: %.4E",
                iteration,
                hpwls.sum().item<float>(),
                overflows[0].item<float>(),
                overflows[1].item<float>(),
                overflows[2].item<float>(),
                obj.item<float>(),
                ps.density_weight,
                ps.wa_coeff);
            // logger.info(
            //     "density_weights: [%.4E, %.4E, %.4E] wa_coeff: [%.4E, %.4E]",
            //     ps.density_weights[0], ps.density_weights[1], ps.density_weights[2],
            //     ps.wa_coeffs[0], ps.wa_coeffs[1]);

            if (st::setting.draw_placement) {
                if (true) {
                    auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(mov_node_pos_all).to(torch::kCPU);
                    auto true_mov_node_pos_all = mov_node_pos_all.to(torch::kCPU) + node_shift;
                    auto info1 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_GP_0");
                    draw_fig_with_cairo_cpp(
                        true_mov_node_pos_all.to(torch::kCPU).index({Slice(cell_mov_lhs, cell_mov_rhs), "..."}),
                        node_size_bot, data, info1);
                    auto info2 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_GP_1");
                    draw_fig_with_cairo_cpp(
                        true_mov_node_pos_all.to(torch::kCPU).index({Slice(cell_mov_lhs, cell_mov_rhs), "..."}),
                        node_size_top, data, info2);
                    auto info3 = make_tuple(st::setting.round_recursion, iteration, message + data.design_name + "_GP_2");
                    draw_fig_with_cairo_cpp(
                        true_mov_node_pos_all.to(torch::kCPU), mov_node_size_all.to(torch::kCPU), data, info3);
                    /* draw vias */
                    auto info = make_tuple(st::setting.round_recursion, iteration, message + data.design_name + "_VIA_GP");
                    draw_fig_with_cairo_cpp(true_mov_node_pos_all.to(torch::kCPU).index({Slice(cell_mov_rhs, mov_rhs)}),
                                            via_mov_node_size,
                                            via_data,
                                            info,
                                            viaColor);
                }
                // saveChannels(init_density_maps, message+to_string(st::setting.round_recursion)+"_"+
                //                                         to_string(iteration)+"_"+data.design_name + "_density_map");
            }
        }
        if (iteration >= st::setting.minGPStep && ps.need_to_early_stop()) {
            break;
        }
    }
    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    /* retrieve best score and evaluate without filler */
    auto [best_sol, best_hpwl, best_overflow, best_iteration] = ps.get_best_solution();
    if (best_iteration > -1) {
        logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f", iteration, best_hpwl, best_overflow);
    } else {
        logger.info("GP Stop! cannot find best solution");
    }
    if (best_sol.numel() != 0) {
        logger.info("Roll back to the best sol");
        mov_node_pos_all.data().copy_(best_sol);
    }
    node_pos = mov_node_pos_all.index({Slice(mov_lhs, mov_rhs)}).contiguous();
    auto [hpwl, overflows] = evaluate_placement_multi_circuit(node_pos, density_map_layers, init_density_maps, data);
    logger.notice("After GP, best solution eval, exact HPWL: %.4E exact Overflow: (%.4f, %.4f, %.4f)",
                  (hpwl).item().toFloat(),
                  overflows[0].item<float>(),
                  overflows[1].item<float>(),
                  overflows[2].item<float>());

    data.to(torch::kCPU);
    via_data.to(torch::kCPU);
    node_pos = node_pos.to(torch::kCPU);

    if (st::setting.visualize_curve) ps.visualize();

    if (st::setting.block_row) {
        data.node_size.index({"...", 1}) /= 0.9;  // TODO: config
        data.node_size_top.index({"...", 1}) /= 0.9;
        data.node_size_bot.index({"...", 1}) /= 0.9;
        data.node_size = torch::cat({data.node_size.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_data.node_size},
                                    0);  // FIXME: original size before expand
    }
    auto node_size = data.node_size.index({Slice(mov_lhs, mov_rhs)}).clone();

    auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
    auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
        evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
    logger.notice("After GP, exact HPWL [bot + top = total / overlap / overhead]: [%f + %f = %f / %f / %f]",
                  hpwl1.item<float>(),
                  hpwl2.item<float>(),
                  (hpwl1 + hpwl2).item<float>(),
                  hpwl_ovlp.item<float>(),
                  hpwl_ovhd.item<float>());

    hpwl_state.hpwls_gp = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl_state.hpwls_gp;
    hpwl_state.log();

    /* draw placement */
    if (true) {
        /* draw cells */
        mov_node_pos = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(torch::kCPU);
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(mov_node_pos).to(torch::kCPU);
        auto true_mov_node_pos = mov_node_pos + node_shift;
        auto info1 = make_tuple(st::setting.round_recursion, 0, message + data.design_name + "_PT_GP_0");
        draw_fig_with_cairo_cpp(true_mov_node_pos, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, message + data.design_name + "_PT_GP_1");
        draw_fig_with_cairo_cpp(true_mov_node_pos, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, message + data.design_name + "_PT_GP_2");
        auto node_pos_draw_cp = torch::cat({true_mov_node_pos, true_mov_node_pos}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
    }
    if (st::setting.round_recursion == 0) {
        via_mov_node_pos = node_pos.index({Slice(cell_mov_rhs, mov_rhs)}).to(torch::kCPU);
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(via_mov_node_pos).to(torch::kCPU);
        auto true_via_mov_node_pos = via_mov_node_pos + node_shift;
        auto via_node_size = via_data.node_size.clone();
        if (st::setting.round_recursion > 0) {
            via_node_size.index_put_({"...", 0}, data.bondingInfo[0] + data.bondingInfo[2]);
            via_node_size.index_put_({"...", 1}, data.bondingInfo[1] + data.bondingInfo[2]);
        }
        /* draw vias */
        auto info = make_tuple(st::setting.round_recursion, iteration, message + data.design_name + "_VIA_GP");
        draw_fig_with_cairo_cpp(true_via_mov_node_pos, via_node_size, via_data, info, viaColor);
    }

    /* save to .pt model */
    if (st::setting.save_model) {
        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string model_dir = res_root.string() + "/gp.pt";
        logger.info("Save model to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
        std::string pt_dir = res_root.string() + "/pt.pt";
        torch::save(data.node_die, pt_dir);
    }

#ifdef DEBUG
    if (st::setting.round_recursion > 0) {
        logger.info("placing vias to optimal region");
        cout << "-------------------------------0\n";
        (via_data.bonding_map == 1).to(device);
        cout << "-------------------------------1\n";
        data.hyperedge_list_end.to(device);
        cout << "-------------------------------2\n";
        data.hyperedge_list.to(device);
        cout << "-------------------------------3\n";
        data.pin_rel_cpos.to(device);
        cout << "-------------------------------4\n";
        data.pin_id2node_id.to(device);
        cout << "-------------------------------5\n";
        data.node_die.to(device);
        cout << "-------------------------------6\n";
        // node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(device);
        cout << "-------------------------------7\n";

        // auto [via_pos, tmp_via] = wa_wirelength_hpwl::place_via_to_optim(
        //     node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(device), node_die.to(device),
        //     data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device), data.hyperedge_list.to(device),
        //     data.hyperedge_list_end.to(device), (via_data.bonding_map == 1).to(device));
        data.reset();
        auto [via_pos, tmp_via] =
            wa_wirelength_hpwl::place_via_to_optim(node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(device),
                                                   node_die.to(device),
                                                   data.pin_id2node_id.to(device),
                                                   data.pin_rel_cpos.to(device),
                                                   data.hyperedge_list.to(device),
                                                   data.hyperedge_list_end.to(device),
                                                   (torch::ones(data.num_nets) == 1).to(device));

        /* via layer */  // FIXME:
        node_pos.index({Slice(cell_mov_rhs, mov_rhs)}).data().copy_(via_pos.data());
        mov_node_size_all.index_put_({Slice(cell_mov_rhs, mov_rhs), 0}, data.bondingInfo[0] + data.bondingInfo[2]);
        mov_node_size_all.index_put_({Slice(cell_mov_rhs, mov_rhs), 1}, data.bondingInfo[1] + data.bondingInfo[2]);
        // node_pos.to(device);
        auto [tmp, density_map_via] =
            density_map_layers[2].direct_calc_overflow(node_pos.to(device), mov_node_size_all, init_density_maps[2]);
        if (true) {
            via_mov_node_pos = via_pos.clone().to(torch::kCPU);
            auto via_node_size = via_data.node_size.clone().to(torch::kCPU);
            /* draw vias */
            auto info = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_VIA_GP_CUT");
            draw_fig_with_cairo_cpp(via_mov_node_pos, via_node_size, via_data, info, viaColor);

            // via_node_size.index_put_({"...", 0}, data.bondingInfo[0] + data.bondingInfo[2]);
            // via_node_size.index_put_({"...", 1}, data.bondingInfo[1] + data.bondingInfo[2]);
            /* draw vias croos chip */
            info = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_VIA_GP");
            draw_fig_with_cairo_cpp(via_mov_node_pos, via_node_size, via_data, info, viaColor);

            auto via_node_size_cut =
                -(via_node_size - torch::tensor({(data.bondingInfo[0] + data.bondingInfo[2]).item<float>(),
                                                 (data.bondingInfo[1] + data.bondingInfo[2]).item<float>()}));
            auto via_node_pos_draw_cp = torch::cat({via_mov_node_pos, via_mov_node_pos}, 0);
            auto via_node_size_draw_cp = torch::cat({via_node_size, via_node_size_cut}, 0);
            draw_fig_with_cairo_cpp_cross_chip(via_node_pos_draw_cp, via_node_size_draw_cp, data, info);
        }

        // create empty matrix for the gaussian kernel
        int sigma = 2;  // standard deviation of the distribution
        int kernel_width = st::setting.kernel_size;
        kernel_width = kernel_width / 2 * 2 + 1;
        logger.info("Via density map aplied with gaussian blur %d", kernel_width);
        torch::Tensor kernel_matrix = torch::zeros({kernel_width, kernel_width}, dtype(torch::kFloat));
        int kernel_half_width = kernel_width / 2;
        for (int i = -kernel_half_width; i <= kernel_half_width; i++)
            for (int j = -kernel_half_width; j <= kernel_half_width; j++) {
                float r = sqrt(i * i + j * j);
                kernel_matrix[i + kernel_half_width][j + kernel_half_width] =
                    exp(-r / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            }
        kernel_matrix /= kernel_matrix.sum();
        auto density_map_via_blur = torch::zeros_like(density_map_via);

        density_map_via_blur = apply_kernel(density_map_via.to(device),
                                            density_map_via_blur.to(device),
                                            density_map_via.size(0),
                                            density_map_via.size(1),
                                            kernel_matrix.to(device),
                                            kernel_width);

        std::filesystem::path eval(std::string("eval"));
        std::filesystem::path fig_root = logger.res_root / eval;
        if (!std::filesystem::exists(fig_root)) {
            std::filesystem::create_directories(fig_root);
        }
        auto density_map = torch::rot90(density_map_via_blur.to(torch::kCPU));
        // plot_pt(density_map,
        //         (char *)"plot",
        //         (char *)fig_root.string().c_str(),
        //         (char *)("density_via.png"));

        torch::Tensor net_grad_weight = torch::zeros({mov_rhs}, dtype(torch::kFloat));
        net_grad_weight.index_put_({Slice(cell_mov_rhs, mov_rhs)}, 1);

        // mov_node_size_all.index_put_({Slice(cell_mov_rhs, mov_rhs), 0}, data.bondingInfo[0] + data.bondingInfo[2]);
        // mov_node_size_all.index_put_({Slice(cell_mov_rhs, mov_rhs), 1}, data.bondingInfo[1] + data.bondingInfo[2]);

        auto mov_node_wgt_grad = density_map_layers[2].density_grad(
            node_pos.to(device), mov_node_size_all, net_grad_weight.to(device), density_map_via_blur);
        auto net_wgt_grad = mov_node_wgt_grad.index({Slice(cell_mov_rhs, mov_rhs)});

        logger.info("%d nets can be placed without overflow, %d nets are already cut.",
                    net_wgt_grad.sum().item<int>(),
                    via_data.bonding_map.sum().item<int>());
        data.net_wgt_grad = net_wgt_grad.to(torch::kCPU);

        st::setting.block_row = 1;
        // exit(1);
    }

#endif  // DEBUG

    if (st::setting.sw) {
        torch::Tensor density_maps =
            torch::zeros({st::setting.num_den_layer, st::setting.num_bin_x, st::setting.num_bin_y},
                         torch::dtype(node_pos.dtype()).device(device));
        for (int i = 0; i < st::setting.num_den_layer; i++) {
            auto [overflow_chip, density_map] = density_map_layers[i].direct_calc_overflow(
                node_pos.to(device), node_size.to(device), init_density_maps[i]);
            density_maps[i] = density_map;
        }
        auto dneisty_map_dif = torch::abs(density_maps[1] - density_maps[0]);
        std::filesystem::path fig_path = logger.res_root;

        int sigma = 2;  // standard deviation of the distribution
        int kernel_width = st::setting.kernel_size;
        kernel_width = kernel_width / 2 * 2 + 1;
        logger.info("Density map aplied with gaussian blur %d", kernel_width);
        // create empty matrix for the gaussian kernel
        torch::Tensor kernel_matrix = torch::zeros({kernel_width, kernel_width}, dtype(torch::kFloat));
        int kernel_half_width = kernel_width / 2;
        for (int i = -kernel_half_width; i <= kernel_half_width; i++) {
            for (int j = -kernel_half_width; j <= kernel_half_width; j++) {
                float r = sqrt(i * i + j * j);
                kernel_matrix[i + kernel_half_width][j + kernel_half_width] =
                    exp(-r / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            }
        }
        kernel_matrix /= kernel_matrix.sum();
        auto dneisty_map_dif_blur = torch::zeros_like(dneisty_map_dif);
        dneisty_map_dif_blur = apply_kernel(dneisty_map_dif.to(device),
                                            dneisty_map_dif_blur.to(device),
                                            dneisty_map_dif_blur.size(0),
                                            dneisty_map_dif_blur.size(1),
                                            kernel_matrix.to(device),
                                            kernel_width);

        std::filesystem::path eval(std::string("eval"));
        std::filesystem::path fig_root = logger.res_root / eval;
        if (!std::filesystem::exists(fig_root)) {
            std::filesystem::create_directories(fig_root);
        }
        auto density_map = torch::rot90(dneisty_map_dif_blur.to(torch::kCPU));
        // plot_pt(density_map,
        //         (char *)"plot",
        //         (char *)fig_root.string().c_str(),
        //         (char *)("density_dif_blur.png"));

        torch::Tensor node_density_weight = torch::ones({mov_rhs}, dtype(torch::kFloat));
        auto mov_node_wgt_grad =
            density_map_layers[0]
                .density_grad(
                    node_pos.to(device), node_size.to(device), node_density_weight.to(device), dneisty_map_dif_blur)
                .index({Slice(cell_mov_lhs, cell_mov_rhs)});

        auto cell_mov_node_weight = torch::_cast_Float(mov_node_wgt_grad > 0.5);

        data.node_wgt_grad = mov_node_wgt_grad.to(torch::kCPU);
        // data.node_pos = torch::empty({0});
    }

    // st::setting.round_recursion--;
    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    return node_pos;
}