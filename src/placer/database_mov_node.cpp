#include "database.h"
#include "gr/ViaData.h"

// TODO: init_method == "randn_center"
tuple<at::Tensor, at::Tensor, at::Tensor> NodeData::get_mov_node_info() {
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor mov_node_pos = node_pos.index({Slice(mov_lhs, mov_rhs)}).clone();
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)}).clone();

    /* core ll/ur offset */
    at::Tensor single_sideline_ll = die_ll - 1e-4;
    at::Tensor single_sideline_ur = die_ur + die_ll - 1e-4;

    // xl0 yl0 xl1 yl1
    torch::Tensor single_sideline_space_ll = torch::tensor({(float)st::setting.sideline, 
                                                        (float)st::setting.sideline}, dtype(torch::kFloat));
    torch::Tensor single_sideline_space_ur = torch::tensor({(float)st::setting.sideline, 
                                                        (float)st::setting.sideline}, dtype(torch::kFloat));

    mov_node_sideline_ll = single_sideline_ll.repeat({num_nodes + __num_fillers__, 1});
    mov_node_sideline_ur = single_sideline_ur.repeat({num_nodes + __num_fillers__, 1});
    mov_node_sideline_ll += single_sideline_space_ll;
    mov_node_sideline_ur -= single_sideline_space_ur;
    // mov_node_sideline_ll = torch::cat({mov_node_sideline_ll, 
    //                             (single_sideline_ll).repeat({__num_fillers__, 1})}, 0);
    // mov_node_sideline_ur = torch::cat({mov_node_sideline_ur, 
    //                             (single_sideline_ur).repeat({__num_fillers__, 1})}, 0);

    at::Tensor scale = (die_ur - die_ll) * 0.001;
    at::Tensor loc = (die_ur + die_ll) * 0.5;
    mov_node_pos = torch::randn_like(mov_node_pos) * scale + loc;
    torch::sort(mov_node_pos, 0, true);
    // FIXME: rand mov node pos
    if (st::setting.use_filler) {
        // TODO: enable fence
        at::Tensor filler_pos = torch::rand({__num_fillers__, 2}, torch::dtype(mov_node_size.dtype()));
        at::Tensor scale = die_ur - die_ll;
        at::Tensor shift = die_ll;
        filler_pos = filler_pos * scale + shift;

        mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
        mov_node_size = torch::cat({mov_node_size, filler_size}, 0);
    }
    if (st::setting.noise_ratio > 0) {
        at::Tensor noise = torch::rand_like(mov_node_pos, torch::dtype(mov_node_size.dtype()));
        torch::sort(noise, 0, true);
        noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
        mov_node_pos += noise;
    }
    // at::Tensor expand_ratio = mov_node_pos.new_ones((mov_node_pos.sizes()[0]));
    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)), torch::dtype(mov_node_size.dtype()));
    if (st::setting.clamp_node) {
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        at::Tensor clamp_mov_node_size = mov_node_size.clamp(unit_len * sqrt(2));
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        mov_node_size = clamp_mov_node_size;
    }
    return make_tuple(mov_node_pos, mov_node_size, expand_ratio);
}  // END MODULE

//---------------------------------------------------------------------

tuple<at::Tensor, at::Tensor, at::Tensor> NodeData::get_mov_node_info_cross_chip(bool init_macro, bool init_stdcell, bool move_macro) {
    auto mov_node_die = node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).clone();
    node_size_exact = node_size.index({Slice(cell_mov_lhs, cell_mov_rhs)}).clone();
    

    // mov_cell_areas = mov_cell_areas.index({Slice(0, 2)});

    // node_size = node_size_bot * (1 - mov_node_die).unsqueeze(1) + node_size_top * mov_node_die.unsqueeze(1);
    // node_size_exact = node_size.clone();

    movable_index = make_tuple(cell_mov_lhs, cell_mov_rhs);

    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    mov_cell_areas[0] = torch::sum((1 - mov_node_die) * torch::prod(node_size_bot, 1));
    mov_cell_areas[1] = torch::sum((mov_node_die) * torch::prod(node_size_top, 1));

    /* split into two chips */
    mov_node_sizes.resize(2);
    vector<int> num_movable_nodes_cc(2);
    idx_cc.resize(2);
    vector<int> node_counts(2, 0);
    mov_node_sizes[0] = node_size_bot;
    mov_node_sizes[1] = node_size_top;
    num_movable_nodes_cc[0] = (1 - mov_node_die).sum().item<int>();
    num_movable_nodes_cc[1] = mov_node_die.sum().item<int>();
    idx_cc[0] = torch::zeros(num_movable_nodes_cc[0], dtype(torch::kInt64));
    idx_cc[1] = torch::zeros(num_movable_nodes_cc[1], dtype(torch::kInt64));

    for (int i = 0; i != num_nodes; i++) {
        int die_idx = mov_node_die[i].item<int>();
        node_size_exact[i][0] = mov_node_sizes[die_idx][i][0];
        node_size_exact[i][1] = mov_node_sizes[die_idx][i][1];
        node_size[i][0] = mov_node_sizes[die_idx][i][0];  // TODO: original node size
        node_size[i][1] = mov_node_sizes[die_idx][i][1];
        idx_cc[die_idx][node_counts[die_idx]] = i;
        node_counts[die_idx]++;
    }

    /* select index */
    vector<torch::Tensor> node_poses(2);
    node_poses[0] = node_pos.index_select(0, idx_cc[0]);
    node_poses[1] = node_pos.index_select(0, idx_cc[1]);
    mov_node_sizes[0] = mov_node_sizes[0].index_select(0, idx_cc[0]);
    mov_node_sizes[1] = mov_node_sizes[1].index_select(0, idx_cc[1]);

    /* core ll/ur offset */
    at::Tensor single_sideline_ll = die_ll - 1e-4;
    at::Tensor single_sideline_ur = die_ur + die_ll - 1e-4;

    // // xl0 yl0 xl1 yl1 case4
    // torch::Tensor single_sideline_space_ll = torch::tensor({(float)500.0, (float)0.0, (float)500.0, (float)0.0}, dtype(torch::kFloat)).reshape({2, 2});
    // torch::Tensor single_sideline_space_ur = torch::tensor({(float)500.0, 500 + (die_ur[1] - numRows[0] * rowHeights[0] - 1e-4).item<float>(), 
    //                                                 (float)500.0, 500 + (die_ur[1] - numRows[1] * rowHeights[1] - 1e-4).item<float>()}, dtype(torch::kFloat)).reshape({2, 2});
    
    // // xl0 yl0 xl1 yl1 case4h
    // torch::Tensor single_sideline_space_ll = torch::tensor({(float)500.0, (float)0.0, (float)200.0, (float)0.0}, dtype(torch::kFloat)).reshape({2, 2});
    // torch::Tensor single_sideline_space_ur = torch::tensor({(float)500.0, 200 + (die_ur[1] - numRows[0] * rowHeights[0] - 1e-4).item<float>(), 
    //                                                 (float)500.0, 200 + (die_ur[1] - numRows[1] * rowHeights[1] - 1e-4).item<float>()}, dtype(torch::kFloat)).reshape({2, 2});
    
    // torch::Tensor single_sideline_space_ll = torch::tensor({(float)st::setting.sideline, (float)0.0, (float)st::setting.sideline, (float)0.0}, dtype(torch::kFloat)).reshape({2, 2});
    torch::Tensor single_sideline_space_ll = torch::tensor({(float)st::setting.sideline, (float)st::setting.sideline, (float)st::setting.sideline, (float)st::setting.sideline}, dtype(torch::kFloat)).reshape({2, 2});
    torch::Tensor single_sideline_space_ur = torch::tensor({(float)st::setting.sideline, (float)st::setting.sideline + (die_ur[1] - numRows[0] * rowHeights[0] - 1e-4).item<float>(), 
                                                    (float)st::setting.sideline, (float)st::setting.sideline + (die_ur[1] - numRows[1] * rowHeights[1] - 1e-4).item<float>()}, dtype(torch::kFloat)).reshape({2, 2});
    
    int checkSideline1 = (die_ur[1] - numRows[0] * rowHeights[0]).item<int>();
    int checkSideline2 = (die_ur[1] - numRows[1] * rowHeights[1]).item<int>();
    mov_node_sideline_ll = single_sideline_ll.repeat({num_nodes, 1});
    mov_node_sideline_ur = single_sideline_ur.repeat({num_nodes, 1});
    if(!move_macro)
    {
        mov_node_sideline_ll += single_sideline_space_ll.index_select(0, mov_node_die)*(1-macro_mask).unsqueeze(1); // FIXME:
        mov_node_sideline_ur -= single_sideline_space_ur.index_select(0, mov_node_die)*(1-macro_mask).unsqueeze(1);
    }
    else{
        mov_node_sideline_ll += single_sideline_space_ll.index_select(0, mov_node_die); // FIXME:
        mov_node_sideline_ur -= single_sideline_space_ur.index_select(0, mov_node_die);//@@
    }

    /* fillers multi chip */
    vector<int> num_fillers_cc(2);
    node_area = torch::prod(node_size, 1).unsqueeze(1);
    at::Tensor die_area = at::prod(die_ur - die_ll);
    vector<torch::Tensor> filler_sizes(2);
    /* precond area info */
    auto [mov_lhs, mov_rhs] = movable_index;
    mov_node_area = node_area.index({Slice(mov_lhs, mov_rhs)});
    mov_node_to_num_pins = node_to_num_pins.index({Slice(mov_lhs, mov_rhs)});
    mov_node_die = torch::_cast_Float(mov_node_die);
    for (int i = 0; i < 2; i++) {
        at::Tensor ori_dmap = (init_density_maps[i] / target_density).sum();
        at::Tensor fixed_node_area = ori_dmap * bin_area;
        // at::Tensor placeable_area = die_area - fixed_node_area; // TODO: placeable area
        // at::Tensor placeable_die_area = at::prod(die_ur - die_ll);
        at::Tensor placeable_die_area = at::prod(die_ur - die_ll 
                        - single_sideline_space_ur[i]);

        logger.info(
            "Placeable area of die/sidelined: [%.4E, %4E]", die_area.item<float>(), 
                                                            placeable_die_area.item<float>());
        at::Tensor placeable_area = placeable_die_area - fixed_node_area;

        /* compute_filler */
        at::Tensor mov_node_size = mov_node_sizes[i];
        at::Tensor mov_cell_area = mov_node_size.numel() ? torch::prod(mov_node_size, 1) : torch::zeros({1, 1});
        at::Tensor mov_node_xsize_order = torch::argsort(node_size_exact.index({"...", 0}));

        at::Tensor filler_size_x = torch::mean(
            node_size_exact.index({"...", 0})
                .index({mov_node_xsize_order.index({Slice(int(num_nodes * 0.05), int(num_nodes * 0.95))})}));
        at::Tensor filler_size_y = rowHeights[i];
        if((filler_size_y>filler_size_x).item<int>()==1)
        {
            filler_size_x=filler_size_y;
        }

        at::Tensor total_filler_area =
            max(target_density * placeable_area - torch::sum(mov_cell_area), at::tensor(0.0));
        single_filler_size =
            at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>()}, torch::dtype(mov_node_size.dtype()));
        num_fillers_cc[i] =
            torch::round(total_filler_area / (filler_size_x * filler_size_y)).item<int>();  // FIXME: round ? floor
        filler_sizes[i] = single_filler_size.repeat({num_fillers_cc[i], 1});
        mov_node_sideline_ll = torch::cat({mov_node_sideline_ll, 
                                        (single_sideline_ll).repeat({num_fillers_cc[i], 1})}, 0);
        mov_node_sideline_ur = torch::cat({mov_node_sideline_ur, 
                                        (single_sideline_ur).repeat({num_fillers_cc[i], 1})}, 0);

        /* compute_precond_var */
        at::Tensor filler_area = torch::prod(filler_sizes[i], 1).unsqueeze(1);
        mov_node_area = torch::cat({mov_node_area, filler_area}, 0);
        at::Tensor filler_to_num_pins = mov_node_to_num_pins.new_zeros({num_fillers_cc[i], 1});
        assert(filler_to_num_pins.sizes() == filler_area.sizes());
        mov_node_to_num_pins = torch::cat({mov_node_to_num_pins, filler_to_num_pins}, 0);
        torch::Tensor filler_die = torch::ones(num_fillers_cc[i], torch::dtype(mov_node_die.dtype())) * i;
        mov_node_die = torch::cat({mov_node_die, filler_die}, 0);

        logger.info("#Fillers: %d Filler size: (%.4e, %.4e)", num_fillers_cc[i], single_filler_size[0].item<double>(),
                    single_filler_size[1].item<double>());
    }

    /* compute_sorted_node_map */
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);
    mov_sorted_map = mov_sorted_map.contiguous();
    auto [tmp1, mov_conn_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
    auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
    mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
    filler_sorted_map = filler_sorted_map.contiguous();
    sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));

    /* mov_node_pos */
    at::Tensor mov_node_pos = node_pos.index({Slice(mov_lhs, mov_rhs)}).clone();
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)}).clone();
    at::Tensor scale = (die_ur - die_ll) * 0.001;
    at::Tensor loc = (die_ur + die_ll) * 0.5;
    if(init_macro)
    {
        mov_node_pos = torch::randn_like(mov_node_pos) * scale + loc;
    }else if(!init_macro && init_stdcell){
        at::Tensor rand_pos = torch::randn_like(mov_node_pos) * scale + loc;
        auto macro_mask_2d = macro_mask.unsqueeze(1);
        mov_node_pos = mov_node_pos * macro_mask_2d + rand_pos * (1-macro_mask_2d);
        mov_node_pos = mov_node_pos.detach();
    }else{
        mov_node_pos = mov_node_pos.detach();
    }

    /* filler_mov_node_pos */
    int total_num_filler = num_fillers_cc[0] + num_fillers_cc[1];
    at::Tensor filler_pos = torch::rand({total_num_filler, 2}, torch::dtype(mov_node_size.dtype()));
    scale = die_ur - die_ll;
    at::Tensor shift = die_ll;
    st::setting.filler_type = st::setting.use_pre_gp ? "surround" : st::setting.filler_type;
    if (st::setting.filler_type != "center") {
        logger.info("using filler type: surrounding...");
        bool flag = false;
        float r = 0.5;
        auto filler_pos_filter = torch::empty({0, 2});
        while(!flag) {
            at::Tensor filler_pos_tmp = torch::rand({total_num_filler, 2}, torch::dtype(mov_node_size.dtype())) - r;
            auto radius = (filler_pos_tmp * filler_pos_tmp).sum(1);

            // auto selector = torch::_cast_Int(radius > ((r - 0.05) * (r - 0.05)));
            auto selector = torch::_cast_Int(radius > (r * r));

            auto idx_selector = torch::zeros(selector.sum().item<int>(), dtype(torch::kInt64));
            int idx_cnt = 0;
            for (int k = 0; k < selector.size(0); k++) {
                if (selector[k].item<int>() == 1) {
                    idx_selector[idx_cnt] = k;
                    idx_cnt++;
                }
            }

            auto filler_pos_cat = filler_pos_tmp.index_select(0, idx_selector);

            filler_pos_filter = torch::cat({filler_pos_filter, filler_pos_cat});
            if (filler_pos_filter.size(0) > total_num_filler) flag = true;
        }
        filler_pos_filter = filler_pos_filter.index({Slice(0, total_num_filler), "..."});
        filler_pos = filler_pos_filter + r;
    }
    filler_pos = filler_pos * scale + shift;

    mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
    mov_node_size = torch::cat({mov_node_size, filler_sizes[0]}, 0);
    mov_node_size = torch::cat({mov_node_size, filler_sizes[1]}, 0);

    /* clamp node */
    at::Tensor noise = torch::rand_like(mov_node_pos);
    noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
    if(init_macro)
    {
        mov_node_pos += noise;
    }else if(!init_macro && init_stdcell){
        auto macro_mask_2d = macro_mask.unsqueeze(1);
        mov_node_pos.slice(0, 0, macro_mask_2d.size(0)) += noise.slice(0, 0, macro_mask_2d.size(0)) * (1 - macro_mask_2d);
        //mov_node_pos += noise*(1-macro_mask_2d);
    }else{
        
    }

    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)), torch::dtype(mov_node_size.dtype()));
    if (st::setting.clamp_node) {
        logger.info("clamp nodes to sqrt2 size of bins");
        // at::Tensor expand_ratio = mov_node_pos.new_ones((mov_node_pos.sizes()[0]));
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        at::Tensor clamp_mov_node_size = mov_node_size.clamp(unit_len * sqrt(2));
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        mov_node_size = clamp_mov_node_size;
    }

    /* updata pin rel pos */
    // for (int i = 0; i < num_nets; ++i) {
    for (int i = 0; i < num_pins; ++i) {//@@@
        int64_t node_id = pin_id2node_id[i].item<long>();
        if (node_die[node_id].item<int>() == 0) {
            pin_rel_cpos[i] = pin_rel_cpos_bot[i];
        } else {
            pin_rel_cpos[i] = pin_rel_cpos_top[i];
        }
    }

    // mov_node_weights.resize(2);
    mov_node_weights = torch::empty({2, mov_node_die.size(0)}, dtype(torch::kFloat));
    mov_node_weights[0] = (1 - mov_node_die.clone());
    mov_node_weights[1] = mov_node_die.clone();
    return make_tuple(mov_node_pos, mov_node_size, expand_ratio);
}  // END MODULE

//---------------------------------------------------------------------

tuple<at::Tensor, at::Tensor, at::Tensor> NodeData::get_mov_node_info_cross_chip_with_via(
    PlaceData& via_data, tuple<torch::Tensor, torch::Tensor, torch::Tensor> mov_node_info,
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> via_mov_node_info) {
    vector<tuple<int, int>> movable_indexes(3);

    auto [mov_lhs, mov_rhs] = movable_index;
    auto [via_mov_lhs, via_mov_rhs] = via_data.movable_index;
    auto [mov_node_pos, mov_node_size, expand_ratio] = mov_node_info;
    auto [via_mov_node_pos, via_mov_node_size, via_expand_ratio] = via_mov_node_info;

    /* update mov_node info */
    auto mov_node_size_all = torch::cat(
        {mov_node_size.index({Slice({mov_lhs, mov_rhs})}), via_mov_node_size.index({Slice({via_mov_lhs, via_mov_rhs})}),
         mov_node_size.index({Slice({mov_rhs, torch::indexing::None})}),
         via_mov_node_size.index({Slice({via_mov_rhs, torch::indexing::None})})},
        0);
    auto mov_node_pos_all = torch::cat(
        {mov_node_pos.index({Slice({mov_lhs, mov_rhs})}), via_mov_node_pos.index({Slice({via_mov_lhs, via_mov_rhs})}),
         mov_node_pos.index({Slice({mov_rhs, torch::indexing::None})}),
         via_mov_node_pos.index({Slice({via_mov_rhs, torch::indexing::None})})},
        0);  // FIXME: which order
    auto expand_ratio_all = torch::cat(
        {expand_ratio.index({Slice({mov_lhs, mov_rhs})}), via_expand_ratio.index({Slice({via_mov_lhs, via_mov_rhs})}),
         expand_ratio.index({Slice({mov_rhs, torch::indexing::None})}),
         via_expand_ratio.index({Slice({via_mov_rhs, torch::indexing::None})})},
        0);

    mov_node_to_num_pins =
        torch::cat({mov_node_to_num_pins.index({Slice({mov_lhs, mov_rhs})}),
                    via_data.mov_node_to_num_pins.index({Slice({via_mov_lhs, via_mov_rhs})}),
                    mov_node_to_num_pins.index({Slice({mov_rhs, torch::indexing::None})}),
                    via_data.mov_node_to_num_pins.index({Slice({via_mov_rhs, torch::indexing::None})})},
                   0);

    /* update node_die: -1 | 0 | 1 | 2 */
    node_die = torch::cat(
        {node_die.index({Slice({mov_lhs, mov_rhs})}), via_data.node_die.index({Slice({via_mov_lhs, via_mov_rhs})})}, 0);

    /* mov_node_weights */
    auto mov_node_weights_tmp = mov_node_weights.clone();
    mov_node_weights = torch::empty({3, mov_node_pos_all.size(0)}, dtype(torch::kFloat));
    mov_node_weights[0] =
        torch::cat({mov_node_weights_tmp.index({0, Slice({mov_lhs, mov_rhs})}),
                    torch::zeros(via_mov_rhs - via_mov_lhs, torch::dtype(torch::kFloat)),
                    mov_node_weights_tmp.index({0, Slice({mov_rhs, torch::indexing::None})}),
                    torch::zeros(via_mov_node_size.size(0) - via_mov_rhs, torch::dtype(torch::kFloat))},
                   0);
    mov_node_weights[1] =
        torch::cat({mov_node_weights_tmp.index({1, Slice({mov_lhs, mov_rhs})}),
                    torch::zeros(via_mov_rhs - via_mov_lhs, torch::dtype(torch::kFloat)),
                    mov_node_weights_tmp.index({1, Slice({mov_rhs, torch::indexing::None})}),
                    torch::zeros(via_mov_node_size.size(0) - via_mov_rhs, torch::dtype(torch::kFloat))},
                   0);
    mov_node_weights[2] =
        torch::cat({torch::zeros(mov_rhs - mov_lhs, torch::dtype(torch::kFloat)),
                    torch::ones(via_mov_rhs - via_mov_lhs, torch::dtype(torch::kFloat)),
                    torch::zeros(mov_node_size.size(0) - mov_rhs, torch::dtype(torch::kFloat)),
                    torch::ones(via_mov_node_size.size(0) - via_mov_rhs, torch::dtype(torch::kFloat))},
                   0);

    /* update pin_id2node_id */
    /* cell_node | via_node */
    // pin_id2node_id -> cells_pin | vias_pin
    torch::Tensor via_id2node_id = torch::arange(num_nodes, num_nodes + num_nets, torch::dtype(torch::kInt64));
    pin_id2node_id = torch::cat({pin_id2node_id, via_id2node_id}, 0);
    torch::Tensor via_id2net_id = torch::arange(0, num_nets, torch::dtype(torch::kInt64));
    pin_id2net_id = torch::cat({pin_id2net_id, via_id2net_id}, 0);
    
    torch::Tensor via_additional_pin_list = torch::arange(num_pins, num_pins + num_nets, torch::dtype(torch::kInt64));
    node2pin_list = torch::cat({node2pin_list, via_additional_pin_list }, 0);
    torch::Tensor via_additional_pin_list_end = torch::arange(num_pins + 1, num_pins + num_nets + 1, torch::dtype(torch::kInt64));
    node2pin_list_end = torch::cat({node2pin_list_end, via_additional_pin_list_end }, 0);

    /* update pin_rel_cpos */
    torch::Tensor via_pin_rel_cpos = torch::zeros({num_nets, 2}, torch::dtype(mov_node_pos.dtype()));
    pin_rel_cpos = torch::cat({pin_rel_cpos, via_pin_rel_cpos}, 0);

    /* hyperlists */
    // hyperedge_list_raw.resize(2, vector<index_type>(0));
    // hyperedge_list_end_raw.resize(2, vector<index_type>(0));
    hyperedge_list_cc.resize(2);
    hyperedge_list_end_cc.resize(2);
    hyperedge_list_raw.clear();
    hyperedge_list_raw.resize(2);
    hyperedge_list_end_raw.clear();
    hyperedge_list_end_raw.resize(2);

    /* tensor accessors */
    const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = pin_id2node_id.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = __ori_hyperedge_list__.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = __ori_hyperedge_list_end__.accessor<int64_t, 1>();
    const torch::TensorAccessor<int, 1> cell_die_at = node_die.accessor<int, 1>();

    /* update multi circuit hyperedge lists */
    vector<index_type> ptrs(2, 0);
    vector<index_type> last_ptrs(2, 0);
    for (index_type i = 0; i < num_nets; ++i) {
        index_type start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end_at[i - 1];
        }
        index_type end_idx = hyperedge_list_end_at[i];
        for (index_type idx = start_idx; idx < end_idx; idx++) {
            index_type node_id = pin_id2node_id_at[idx];
            int chip_id = cell_die_at[node_id];
            // hyperedge_list_raw[chip_id].push_back(pin_id2pin_id_cc[idx]); // FIXME:
            hyperedge_list_raw[chip_id].push_back(idx);
            ptrs[chip_id]++;
        }

        /* if a net is cut */
        if (via_data.node_die[i].item<int>() == 2) {
            for (int j = 0; j < 2; j++) {
                // hyperedge_list_raw[j].push_back(pin_id2node_id_raw[j].size() + i);
                hyperedge_list_raw[j].push_back(num_pins + i);  // FIXME:
                ptrs[j]++;
            }
        }

        /* add hyperedge_list_end */
        for (int j = 0; j < 2; j++) {
            if (ptrs[j] != last_ptrs[j]) {
                hyperedge_list_end_raw[j].push_back(ptrs[j]);
            }
            last_ptrs[j] = ptrs[j];
        }
    }
    /* dump to 2 chips */
    for (int i = 0; i < 2; i++) {
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        index_type num_blob = hyperedge_list_raw[i].size();
        hyperedge_list_cc[i] = torch::from_blob(hyperedge_list_raw[i].data(), {num_blob}, options).clone();
        num_blob = hyperedge_list_end_raw[i].size();
        hyperedge_list_end_cc[i] = torch::from_blob(hyperedge_list_end_raw[i].data(), {num_blob}, options).clone();
    }


    // pin_id2node_id = torch::zeros(num_pins + num_nets, torch::dtype(pin_id2node_id.dtype()));
    hyperedge_list = torch::zeros(num_pins + num_nets, torch::dtype(hyperedge_list.dtype()));
    hyperedge_list_end = torch::zeros_like(hyperedge_list_end);
    /* update hyperedge lists */
    index_type ptr = 0;
    index_type last_idx = 0;
    for (int i = 0; i < num_nets; ++i) {
        index_type start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end_at[i - 1];
        }
        index_type end_idx = hyperedge_list_end_at[i];
        /* add cell pin */
        for (index_type idx = start_idx; idx < end_idx; idx++) {
            hyperedge_list[ptr] = hyperedge_list_at[idx];
            ptr++;
        }
        /* add via */
        hyperedge_list[ptr] = num_pins + i;
        ptr++;
        last_idx += end_idx - start_idx + 1;
        hyperedge_list_end[i] = last_idx;
    }

    /* update node area info | sorted maps */
    mov_cell_areas = torch::cat({mov_cell_areas, torch::sum(via_data.mov_cell_area, 0).unsqueeze(0)}, 0);
    movable_index = make_tuple(mov_lhs, mov_rhs + via_mov_rhs);
    tie(mov_lhs, mov_rhs) = movable_index;
    /* compute_sorted_node_map */
    // TODO: for parameter_scheduler
    // node_size = mov_node_size_all.index({Slice({cell_mov_lhs, cell_mov_rhs})}).clone();
    node_size = torch::cat({node_size, via_data.node_size}, 0);  // FIXME: original size before expand
    mov_node_area = torch::prod(mov_node_size_all, 1).unsqueeze(1);
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);
    mov_sorted_map = mov_sorted_map.contiguous();
    auto [tmp1, mov_conn_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
    auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
    mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
    filler_sorted_map = filler_sorted_map.contiguous();
    sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));

    mov_node_sideline_ll = torch::cat({mov_node_sideline_ll, via_data.mov_node_sideline_ll}, 0);
    mov_node_sideline_ur = torch::cat({mov_node_sideline_ur, via_data.mov_node_sideline_ur}, 0);

    return make_tuple(mov_node_pos_all, mov_node_size_all, expand_ratio_all);
}
