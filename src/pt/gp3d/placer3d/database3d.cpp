

#include "database3d.h"

NodeData3D::NodeData3D(NodeData& data) {
    device = data.device;
    // num_bin_x = st::setting.num_bin_x / st::setting.bin_expand;
    // num_bin_y = st::setting.num_bin_y / st::setting.bin_expand;
    num_bin_x = st::setting.num_bin_3d;
    num_bin_y = st::setting.num_bin_3d;
    num_bin_z = st::setting.num_bin_z;  // TODO:  // FIXME:

    /* construct ffom placedb */
    movable_index = data.movable_index;
    connected_index = data.connected_index;
    fixed_index = data.fixed_index;
    node_type_indices = data.node_type_indices;

    /* copy form data */
    die_info = data.die_info.clone();  // TODO: ori_die_info / die_info
    core_info = data.core_info.clone();
    node_size = data.node_size.clone();
    pin_rel_cpos = data.pin_rel_cpos.clone();
    macro_mask = data.macro_mask.clone();

    __ori_die_lx__ = die_info[0].item<int>();
    __ori_die_hx__ = die_info[1].item<int>();
    __ori_die_ly__ = die_info[2].item<int>();
    __ori_die_hy__ = die_info[3].item<int>();
    __ori_die_lz__ = 0;  // TODO:
    __ori_die_hz__ = (__ori_die_hx__ + __ori_die_hy__) / 25;
    // if(st::setting.mode==4)
    // {
    //     __ori_die_hz__ = (__ori_die_hx__ + __ori_die_hy__) / 20;
    // }
    // else{
    //     __ori_die_hz__ = (__ori_die_hx__ + __ori_die_hy__) / 2;
    // }
    // __ori_die_hz__ = 2;

    site_width = data.site_width;
    site_height = data.site_height;
    site_depth = __ori_die_hz__ / num_bin_z;

    // TODO: 2.0
    die_info = torch::cat({die_info, torch::tensor({0.0, (double)__ori_die_hz__}, torch::dtype(die_info.dtype()))}, 0);
    core_info =
        torch::cat({core_info, torch::tensor({0.0, (double)__ori_die_hz__}, torch::dtype(die_info.dtype()))}, 0);

    pin_id2node_id = data.pin_id2node_id.clone();
    hyperedge_index = data.hyperedge_index.clone();
    hyperedge_list = data.hyperedge_list.clone();
    hyperedge_list_end = data.hyperedge_list_end.clone();
    pin_id2net_id = data.pin_id2net_id.clone();
    node2pin_list = data.node2pin_list.clone();
    node2pin_list_end = data.node2pin_list_end.clone();
    node_die = st::setting.use_pre_pt ? data.node_die.clone() : node_die;
    // node_pos = data.node_pos.clone();

    rowHeights = data.rowHeights.clone();
    maxUtilM = data.maxUtilM.clone();
    actualUtilM = data.actualUtilM.clone();
    mov_cell_util = data.mov_cell_util.clone();
    bondingInfo = data.bondingInfo.clone();
    std::tie(cell_mov_lhs, cell_mov_rhs) = movable_index;

    /* node info */
    num_nodes = data.num_nodes;
    num_pins = data.num_pins;
    num_nets = data.num_nets;
    node_size = data.node_size_top.clone();  // FIXME:
    if (shrink_size < 1) {
        node_size = data.node_size.clone();  // FIXME:
    }
    // node_size = data.node_size.clone();  // FIXME:

    pin_rel_cpos = data.pin_rel_cpos.clone();
    pin_rel_cpos_top = data.pin_rel_cpos_top.clone();
    pin_rel_cpos_bot = data.pin_rel_cpos_bot.clone();
    pin_size = data.pin_size.clone();
    mov_node_weights.resize(2);

    /* bin info */
    // clamp_node = st::setting.num_bin_3d == 1 ? st::setting.clamp_node : false;
    clamp_node = st::setting.clamp_node;
    shrink_size = st::setting.shrink_size;
    target_density = 1 / shrink_size;  // TODO:

    logger.info("Cells are shrunk by %.2f", shrink_size);

    /* expand dimension */
    // FIXME:node_size
    auto node_size_z = __ori_die_hz__ / 2 / shrink_size * torch::ones({num_nodes, 1}, torch::dtype(node_size.dtype()));
    // auto node_size_z = torch::mean(node_size) * torch::ones({num_nodes, 1}, torch::dtype(node_size.dtype()));

    node_size = torch::cat({node_size, node_size_z}, 1);
    node_pos = torch::randn_like(node_size);
    auto pin_rel_cpos_z = torch::zeros({num_pins, 1}, torch::dtype(pin_rel_cpos.dtype()));
    pin_rel_cpos = torch::cat({pin_rel_cpos, pin_rel_cpos_z}, 1);
    pin_rel_cpos_top = torch::cat({pin_rel_cpos_top, pin_rel_cpos_z}, 1);
    pin_rel_cpos_bot = torch::cat({pin_rel_cpos_bot, pin_rel_cpos_z}, 1);
    init_density_map = torch::zeros({num_bin_x, num_bin_y, num_bin_z}, torch::dtype(node_size.dtype()));

    logger.info("Tech/Node/Row Ratio: %.2f, %.2f, %.2f",
                data.tech_ratio.item<float>(),
                data.node_ratio.item<float>(),
                (data.rowHeights[0] / data.rowHeights[1]).item<float>());

    node_util_weight = torch::tensor({data.node_ratio.item<float>(), (float)1.0}, torch::dtype(node_size.dtype()));

    node_util_weight_y = torch::tensor({(data.rowHeights[0] / data.rowHeights[1]).item<float>(), (float)1.0},
                                       torch::dtype(node_size.dtype()));

    logger.info("Top size x %.2f for balance", 1 + st::setting.top_util_filler);

    node_util_weight_x = torch::tensor({(data.node_ratio / (data.rowHeights[0] / data.rowHeights[1])).item<float>(),
                                        (float)1.0 * (float)(1 + st::setting.top_util_filler)},
                                       torch::dtype(node_size.dtype()));
    
    ratio_difference = data.node_size_bot/data.node_size_top;

    if (shrink_size < 1) {
        node_util_weight = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
        node_util_weight_y = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
        node_util_weight_x = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
    }

    __die_shift__ = torch::cat({data.__die_shift__, torch::tensor({0.0})}, 0);
    __die_scale__ = torch::cat({data.__die_scale__, data.__die_scale__.slice(0, 0, 1)}, 0);

    // node_util_weight = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
    // node_util_weight_y = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
    // node_util_weight_x = torch::tensor({(float)1.0, (float)1.0}, torch::dtype(node_size.dtype()));
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::backup_ori_var() {


    /* net info */
    __ori_pin_id2node_id__ = pin_id2node_id.clone();
    __ori_hyperedge_list__ = hyperedge_list.clone();
    __ori_hyperedge_list_end__ = hyperedge_list_end.clone();
    __ori_pin_rel_cpos__ = pin_rel_cpos.clone();
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::prescale_by_site_width() {
    // inplace scaling
    die_info /= site_width;
    core_info /= site_width;
    node_size /= site_width;
    pin_rel_cpos /= site_width;
    __die_scale__ *= site_width;
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::pre_compute_var() {
    // die related
    float lx = die_info[0].item<float>();
    float hx = die_info[1].item<float>();
    float ly = die_info[2].item<float>();
    float hy = die_info[3].item<float>();
    float lz = die_info[4].item<float>();
    float hz = die_info[5].item<float>();

    unit_len =
        torch::tensor({(hx - lx) / double(num_bin_x), (hy - ly) / double(num_bin_y), (hz - lz) / double(num_bin_z)},
                      torch::dtype(torch::kFloat32));

    die_ur = die_info.reshape({3, 2}).t()[1].clone();
    die_ll = die_info.reshape({3, 2}).t()[0].clone();
    core_ur = core_info.reshape({3, 2}).t()[1].clone();
    core_ll = core_info.reshape({3, 2}).t()[0].clone();
    hpwl_scale = __die_scale__ / site_width;
    // node related
    node_area = torch::prod(node_size.index({"...", Slice(0, 2)}), 1).unsqueeze(1);
    node_to_num_pins = torch::zeros(num_nodes);
    torch::Tensor v = torch::ones(pin_id2node_id.sizes()[0]);
    node_to_num_pins.scatter_add_(0, pin_id2node_id, v);
    node_to_num_pins.unsqueeze_(1);
    // net related
    torch::Tensor start_idx = hyperedge_list_end.roll(1);
    start_idx[0] = 0;
    net_to_num_pins = hyperedge_list_end - start_idx;
    net_mask = torch::logical_and(net_to_num_pins <= st::setting.ignore_net_degree,
                                  net_to_num_pins >= 2);  // 0: ignore, 1: consider in wirelength calculation

    int view = 20;
    torch::Tensor net_idx_plot = torch::arange(0, view, torch::dtype(torch::kFloat));
    torch::Tensor net_weight_plot;
    if (st::setting.net_weight_type == "step") {
        logger.info("Net weight step func, nets less than %d will be applied with wa force %.3f",
                    st::setting.cut_net_thres,
                    -st::setting.net_weight_offset);
        net_weight = (torch::_cast_Float(torch::logical_and(net_to_num_pins <= st::setting.ignore_net_degree,
                                                            net_to_num_pins >= st::setting.cut_net_thres)) -
                      st::setting.net_weight_offset);

        // for (int i = 0; i < num_nets; i++) 
        //     if (net_to_num_pins[i].item<int>() == 2) 
        //         net_weight[i] = 0;

        net_weight_plot = 1 * (torch::_cast_Float(net_idx_plot >= st::setting.cut_net_thres) -
                                                         st::setting.net_weight_offset);
    } else if (st::setting.net_weight_type == "exp") {
        logger.info("Net weight exp func, thrs %d", st::setting.cut_net_thres);
        // net_weight = torch::exp((net_to_num_pins - st::setting.cut_net_thres) *
        // st::setting.omni_float).clamp(0.0, 1.0); net_weight = torch::exp(net_to_num_pins /
        // st::setting.cut_net_thres);
        net_weight = (torch::pow(net_to_num_pins / (float)st::setting.cut_net_thres, st::setting.net_weight_offset) *
                      st::setting.net_weight_coef)
                         .clamp(0.0, 1.0);

        net_weight_plot = (torch::pow(net_idx_plot / (float)st::setting.cut_net_thres, st::setting.net_weight_offset) *
                           st::setting.net_weight_coef)
                              .clamp(0.0, 1.0);
    }

    // obj related
    auto [mov_lhs, mov_rhs] = movable_index;
    mov_cell_area = torch::prod(node_size.index({torch::indexing::Slice(mov_lhs, mov_rhs), "..."}), 1);
    bin_area = torch::prod(unit_len).item<float>();

    at::Tensor die_area = at::prod(die_ur - die_ll) / shrink_size;
    total_mov_cell_areas = torch::zeros({2}, torch::dtype(mov_cell_area.dtype()));
    __total_mov_area_without_filler__ = 0;
    for (int i = 0; i < 2; i++) {
        total_mov_cell_areas[i] = (die_area / 2 * (actualUtilM[i]));
        __total_mov_area_without_filler__ += total_mov_cell_areas[i].item<float>();  // FIXME:
    }
    logger.info("Maximum utilization, cell area[%.4E] + [%.4E] vs die util[%.4E]",
                total_mov_cell_areas[0].item<float>(),
                total_mov_cell_areas[1].item<float>(),
                die_area.item<float>());
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::logging_statistics() {
    logger.info("===================");
    logger.info("#nodes = %d, #nets = %d, #pins = %d", num_nodes, num_nets, num_pins);
    int num_conmov_nodes = get<1>(node_type_indices[0]) - get<0>(node_type_indices[0]);
    int num_fltmov_nodes = get<1>(node_type_indices[1]) - get<0>(node_type_indices[1]);
    int num_confix_nodes = get<1>(node_type_indices[2]) - get<0>(node_type_indices[2]);
    int num_fltfix_nodes = get<1>(node_type_indices[6]) - get<0>(node_type_indices[6]);
    int num_coniopin = get<1>(node_type_indices[3]) - get<0>(node_type_indices[3]);
    int num_fltiopin = get<1>(node_type_indices[5]) - get<0>(node_type_indices[5]);
    int num_blkg = get<1>(node_type_indices[4]) - get<0>(node_type_indices[4]);
    logger.info("#ConnMov = %d, #FloatMov = %d, #ConnFix = %d, #FloatFix = %d, #ConnIOPin = %d, #FloatIOPin = %d",
                num_conmov_nodes,
                num_fltmov_nodes,
                num_confix_nodes,
                num_fltfix_nodes,
                num_coniopin,
                num_fltiopin);
    int lx = die_info[0].item<int>();
    int hx = die_info[1].item<int>();
    int ly = die_info[2].item<int>();
    int hy = die_info[3].item<int>();
    int lz = die_info[4].item<int>();
    int hz = die_info[5].item<int>();
    logger.info("Die Info %d %d %d %d %d %d", lx, hx, ly, hy, lz, hz);
    logger.info("Core Info %d %d %d %d %d %d",
                core_info[0].item<int>(),
                core_info[1].item<int>(),
                core_info[2].item<int>(),
                core_info[3].item<int>(),
                core_info[4].item<int>(),
                core_info[5].item<int>());
    logger.info("Site Width = %d, Row Height = %d", site_width, site_height);
    logger.info("#Bins = (%d, %d, %d), UnitLen = (%.2f, %.2f, %.2f)",
                num_bin_x,
                num_bin_y,
                num_bin_z,
                unit_len[0].item<float>(),
                unit_len[1].item<float>(),
                unit_len[2].item<float>());
    logger.info("target density = %.2f", target_density);
    logger.info("cell area = %.4E", __total_mov_area_without_filler__);
    logger.info("node[0] = (%.2f, %.2f, %.2f)",
                node_size[0][0].item<float>(),
                node_size[0][1].item<float>(),
                node_size[0][2].item<float>());
    logger.info("===================");
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::preprocess() {
    backup_ori_var();
    // prescale_by_site_width();
    pre_compute_var();
    logging_statistics();
}  // END MODULE

void NodeData3D::compute_filler() {
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)});
    at::Tensor die_area = at::prod(die_ur - die_ll) / shrink_size;

    at::Tensor single_sideline_ll = die_ll - 1e-4;
    at::Tensor single_sideline_ur = die_ur + die_ll - 1e-4;

    sidelines_ll = single_sideline_ll.repeat({num_nodes, 1});
    sidelines_ur = single_sideline_ur.repeat({num_nodes, 1});


    __num_fillers__ = 0;
    use_filler = st::setting.use_filler_3d;
    if (use_filler) {
        int num_fillers_single;
        vector<torch::Tensor> filler_sizes(2);

        int utilm_h_idx = torch::argmax(actualUtilM).item<int>();
        // init_density_map already multiplied by args.target_density
        at::Tensor mov_cell_area = torch::prod(mov_node_size, 1);
        int num_movable_nodes = mov_rhs - mov_lhs;
        at::Tensor mov_node_xsize_order = torch::argsort(mov_node_size.index({"...", 0}));

        at::Tensor filler_size_x =
            torch::mean(mov_node_size.index({"...", 0})
                            .index({mov_node_xsize_order.index(
                                {Slice(int(num_movable_nodes * 0.05), int(num_movable_nodes * 0.95))})}));
        at::Tensor filler_size_y = rowHeights.mean();
        at::Tensor filler_size_z = die_ur[2] / shrink_size;
        // if(st::setting.half_filler_height)
        // {
        //     filler_size_z = die_ur[2] / shrink_size / 2;
        // }
        at::Tensor total_filler_area = die_area * (1 - actualUtilM[utilm_h_idx]);

        single_filler_size =
            at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>(), filler_size_z.item<float>()},
                       torch::dtype(mov_node_size.dtype()));
        num_fillers_single =
            torch::round(total_filler_area / (filler_size_x * filler_size_y * filler_size_z)).item<int>();
        
        num_fillers_single *= 0.90;
        
        num_fillers_single_chip = num_fillers_single;

        single_sideline_ll = die_ll.clone();
        single_sideline_ur = die_ur.clone();
        if (num_fillers_single >= 0) {
            // if (true) {
            // __num_fillers__ += num_fillers_single / 4;
            // sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single / 4, 1})}, 0);
            // sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single / 4, 1})}, 0);

            // filler_sizes[0] = single_filler_size.repeat({num_fillers_single / 4, 1});

            // __num_fillers__ += num_fillers_single / 4 * 3 + num_fillers_single / 4 * 3;
            // single_filler_size[2] = filler_size_z / 2;

            // single_sideline_ll[2] = __ori_die_hz__ / 2 * 0;
            // single_sideline_ur[2] = __ori_die_hz__ / 2 * (1 + 0);
            // sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single / 4 * 3, 1})}, 0);
            // sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single / 4 * 3, 1})}, 0);

            // single_sideline_ll[2] = __ori_die_hz__ / 2 * 1;
            // single_sideline_ur[2] = __ori_die_hz__ / 2 * (1 + 1);
            // sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single / 4 * 3, 1})}, 0);
            // sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single / 4 * 3, 1})}, 0);

            // auto single_filler_size0 = single_filler_size.repeat({num_fillers_single / 4 * 3, 1});
            // auto single_filler_size1 = single_filler_size.repeat({num_fillers_single / 4 * 3, 1});

            // filler_sizes[0] = torch::cat({filler_sizes[0], single_filler_size0, single_filler_size1}, 0);
            // filler_size = filler_sizes[0];

            if(!st::setting.half_filler_height) {
                __num_fillers__ += num_fillers_single;

                sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single, 1})}, 0);
                sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single, 1})}, 0);

                filler_sizes[0] = single_filler_size.repeat({num_fillers_single, 1});
                filler_size = filler_sizes[0];

                logger.info("#Cross-Chip-Fillers: %d, size: (%.4e, %.4e, %.4e)",
                            num_fillers_single,
                            single_filler_size[0].item<float>(),
                            single_filler_size[1].item<float>(),
                            single_filler_size[2].item<float>());
            } else {
                __num_fillers__ += 2 * num_fillers_single;
                single_filler_size[2] = filler_size_z / 2;

                single_sideline_ll[2] = __ori_die_hz__ / 2 * 0;
                single_sideline_ur[2] = __ori_die_hz__ / 2 * (1 + 0);
                sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single, 1})}, 0);
                sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single, 1})}, 0);

                single_sideline_ll[2] = __ori_die_hz__ / 2 * 1;
                single_sideline_ur[2] = __ori_die_hz__ / 2 * (1 + 1);
                sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single, 1})}, 0);
                sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single, 1})}, 0);

                auto single_filler_size0 = single_filler_size.repeat({num_fillers_single, 1});
                auto single_filler_size1 = single_filler_size.repeat({num_fillers_single, 1});

                filler_sizes[0] = torch::cat({single_filler_size0, single_filler_size1}, 0);
                filler_size = filler_sizes[0];

                logger.info("#Single-Chip-Fillers: 2 x %d, size: (%.4e, %.4e, %.4e)",
                            num_fillers_single,
                            single_filler_size[0].item<float>(),
                            single_filler_size[1].item<float>(),
                            single_filler_size[2].item<float>());
            }
        }
        int utilm_l_idx = 1 - utilm_h_idx;
        if ((actualUtilM[utilm_h_idx] - actualUtilM[utilm_l_idx]).item<float>() > 1e-6) {
            // filler_size_y = rowHeights[utilm_l_idx] / __die_scale__[1];    // TODO: mean row height
            filler_size_z = die_ur[2] / 2 / shrink_size;  // TODO: filler size z
            total_filler_area = die_area / 2 * (actualUtilM[utilm_h_idx] - actualUtilM[utilm_l_idx]);  // TODO:

            single_filler_size =
                at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>(), filler_size_z.item<float>()},
                           torch::dtype(mov_node_size.dtype()));
            num_fillers_single =
                torch::round(total_filler_area / (filler_size_x * filler_size_y * filler_size_z)).item<int>();

            single_sideline_ll[2] = __ori_die_hz__ / 2 * utilm_l_idx;        // TODO:
            single_sideline_ur[2] = __ori_die_hz__ / 2 * (1 + utilm_l_idx);  // TODO:

            __num_fillers__ += num_fillers_single;
            sidelines_ll = torch::cat({sidelines_ll, single_sideline_ll.repeat({num_fillers_single, 1})}, 0);
            sidelines_ur = torch::cat({sidelines_ur, single_sideline_ur.repeat({num_fillers_single, 1})}, 0);

            filler_sizes[1] = single_filler_size.repeat({num_fillers_single, 1});
            filler_size = torch::cat({filler_size, filler_sizes[1]}, 0);

            num_fillers_cross_chip = num_fillers_single;

            logger.info("#Single-Chip-Fillers: %d, size: (%.4e, %.4e, %.4e)",
                        num_fillers_single,
                        single_filler_size[0].item<float>(),
                        single_filler_size[1].item<float>(),
                        single_filler_size[2].item<float>());
        }
    } else {
        logger.info("disable fillers, cell area[%.4E] vs die util[%.4E]",
                    torch::sum(torch::prod(mov_node_size, 1)).item<float>(),
                    die_area.item<float>());
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::compute_precond_var() {
    auto [mov_lhs, mov_rhs] = movable_index;
    mov_node_area = node_area.index({Slice(mov_lhs, mov_rhs)});

    mov_node_to_num_pins = node_to_num_pins.index({Slice(mov_lhs, mov_rhs)});
    if (use_filler) {
        at::Tensor filler_area = torch::prod(filler_size.index({"...", Slice(0, 2)}), 1).unsqueeze(1);
        mov_node_area = torch::cat({mov_node_area, filler_area}, 0);
        at::Tensor filler_to_num_pins = mov_node_to_num_pins.new_zeros({__num_fillers__, 1});
        assert(filler_to_num_pins.sizes() == filler_area.sizes());
        mov_node_to_num_pins = torch::cat({mov_node_to_num_pins, filler_to_num_pins}, 0);
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::compute_sorted_node_map() {
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);

    mov_sorted_map = mov_sorted_map.contiguous();
    at::Tensor mov_conn_sorted_map = mov_sorted_map;
    if (use_filler) {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        filler_sorted_map = filler_sorted_map.contiguous();
        sorted_maps =
            make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));
    } else {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        at::Tensor filler_sorted_map;
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map);
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData3D::init_filler() {
    logger.info("Processing fillers");
    compute_filler();
    compute_precond_var();
    compute_sorted_node_map();
}  // END MODULE

//---------------------------------------------------------------------

tuple<at::Tensor, at::Tensor, at::Tensor> NodeData3D::get_mov_node_info() {
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)}).clone();

    at::Tensor scale = (die_ur - die_ll) * 0.001;
    scale[2] *= 1;
    at::Tensor loc = (die_ur + die_ll) * 0.5;
    at::Tensor mov_node_pos = torch::randn_like(mov_node_size) * scale + loc;

    // FIXME: rand mov node pos
    if (use_filler) {
        // TODO: enable fence
        at::Tensor filler_pos = torch::rand({__num_fillers__, 3}, torch::dtype(mov_node_size.dtype()));
        at::Tensor scale = die_ur - die_ll;
        at::Tensor shift = die_ll;
        // st::setting.filler_type = (st::setting.force_coeff_2d) ? "surround" : st::setting.filler_type;
        auto filler_type = st::setting.filler_type;
        if (filler_type != "center") {
            logger.info("using filler type: surrounding...");
            bool flag = false;
            float r = 0.5;
            auto filler_pos_filter = torch::empty({0, 2});
            while(!flag) {
                at::Tensor filler_pos_tmp = torch::rand({__num_fillers__, 2}, torch::dtype(mov_node_size.dtype())) - r;
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
                if (filler_pos_filter.size(0) > __num_fillers__) flag = true;
            }
            filler_pos_filter = filler_pos_filter.index({Slice(0, __num_fillers__), "..."});
            auto filler_pos_2d = filler_pos_filter + r;
            filler_pos.index({"...", Slice(0, 2)}).data().copy_(filler_pos_2d.data());
        }
        filler_pos = filler_pos * scale + shift;
        // auto sidelines_ll_filler_z = sidelines_ll.index({torch::indexing::Slice(num_nodes), torch::indexing::Slice(2)});
        // auto sidelines_ur_filler_z = sidelines_ur.index({torch::indexing::Slice(num_nodes), torch::indexing::Slice(2)});
        // auto filler_pos_z = (sidelines_ll_filler_z + sidelines_ur_filler_z) / 2;
        // auto filler_pos_new = torch::cat({filler_pos.index({"...", torch::indexing::Slice(0, 2)}), filler_pos_z}, 1);
        mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
        mov_node_size = torch::cat({mov_node_size, filler_size}, 0);
    }
    if (st::setting.noise_ratio > 0) {
        at::Tensor noise = torch::rand_like(mov_node_pos, torch::dtype(mov_node_size.dtype()));
        noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
        mov_node_pos += noise;
    }

    at::Tensor die_area = at::prod(die_ur - die_ll) / shrink_size;
    logger.info("cell area[%.4E] vs die util[%.4E]",
                torch::sum(torch::prod(mov_node_size, 1)).item<float>(),
                die_area.item<float>());

    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)), torch::dtype(mov_node_size.dtype()));
    if (clamp_node) {
        // logger.info("Cells clamped to bin size on x/y direction");
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        auto clamp_size = unit_len * sqrt(2);
        clamp_size[2] = unit_len[2];
        // at::Tensor clamp_mov_node_size = mov_node_size.clamp(clamp_size);

        logger.info("Cells doubled the size on x/y direction");
        at::Tensor clamp_mov_node_size = mov_node_size.clone();
        clamp_mov_node_size.index({"...", Slice(0, 2)}) *= st::setting.kernel_size;
        if (st::setting.force_coeff_2d == 1) clamp_mov_node_size = mov_node_size.clamp(clamp_size);
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        mov_node_size = clamp_mov_node_size;
    }

    mov_node_weight = torch::ones({mov_node_pos.size(0)}, dtype(mov_node_pos.dtype()));
    mov_node_weights[0] = mov_node_weight;
    mov_node_weights[1] = 1 - mov_node_weight;
    node_die = torch::ones({mov_node_pos.size(0)}, dtype(torch::kInt));

    return make_tuple(mov_node_pos, mov_node_size, expand_ratio);
}  // END MODULE

//---------------------------------------------------------------------

/// follow the PlaceData
/// @param hyperedge_list
/// @param hyperedge_list_end
/// @param pin_id2node_id
/// @param node_size
/// @param mov_node_area
/// @param mov_node_to_num_pins
/// @param movable_index

/// @param mov_node_pos
/// @param mov_node_size
/// @param expand_ratio
/// @param sorted_maps

/// @param sidelines_ll
/// @param sidelines_ur
/// @param mov_node_weight

tuple<at::Tensor, at::Tensor, at::Tensor> NodeData3D::get_mov_node_info_with_via() {
    logger.info("Get mov node info with vias");
    /* update hyperedge lists */
    /* via_mov_node | cell_node */
    // pin_id2node_id -> cells_pin | vias_pin
    // hyperedge_list -> net_pin | via
    // hyperedge_list_end -> hyperedge_list_end + 1
    const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = pin_id2node_id.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = __ori_hyperedge_list__.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = __ori_hyperedge_list_end__.accessor<int64_t, 1>();

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
    /* update pin_id2node_id */
    /* cell_node | via_node */
    // pin_id2node_id -> cells_pin | vias_pin
    torch::Tensor via_id2node_id = torch::arange(num_nodes, num_nodes + num_nets, torch::dtype(torch::kInt64));
    pin_id2node_id = torch::cat({pin_id2node_id, via_id2node_id}, 0);
    torch::Tensor via_id2net_id = torch::arange(0, num_nets, torch::dtype(torch::kInt64));
    pin_id2net_id = torch::cat({pin_id2net_id, via_id2net_id}, 0);

    /// @brief node
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)}).clone();

    at::Tensor scale = (die_ur - die_ll) * 0.001;
    scale[2] *= 1;
    at::Tensor loc = (die_ur + die_ll) * 0.5;
    at::Tensor mov_node_pos = torch::randn_like(mov_node_size) * scale + loc;
    // FIXME: rand mov node pos
    if (use_filler) {
        // TODO: enable fence
        at::Tensor filler_pos = torch::rand({__num_fillers__, 3}, torch::dtype(mov_node_size.dtype()));
        at::Tensor scale = die_ur - die_ll;
        at::Tensor shift = die_ll;
        filler_pos = filler_pos * scale + shift;

        mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
        mov_node_size = torch::cat({mov_node_size, filler_size}, 0);
    }
    if (st::setting.noise_ratio > 0) {
        at::Tensor noise = torch::rand_like(mov_node_pos, torch::dtype(mov_node_size.dtype()));
        noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
        mov_node_pos += noise;
    }

    at::Tensor die_area = at::prod(die_ur - die_ll) / shrink_size;
    logger.info("cell area[%.4E] vs die util[%.4E]",
                torch::sum(torch::prod(mov_node_size, 1)).item<float>(),
                die_area.item<float>());

    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)), torch::dtype(mov_node_size.dtype()));
    if (clamp_node) {
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        at::Tensor clamp_mov_node_size = mov_node_size.clamp(unit_len * sqrt(2));
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        mov_node_size = clamp_mov_node_size;
    }

    mov_node_weight = torch::ones({mov_node_pos.size(0)}, dtype(mov_node_pos.dtype()));
    mov_node_weights[0] = mov_node_weight;
    mov_node_weights[1] = 1 - mov_node_weight;
    node_die = torch::ones({mov_node_pos.size(0)}, dtype(torch::kInt));

    // return make_tuple(mov_node_pos, mov_node_size, expand_ratio);

    /// @brief via
    at::Tensor single_sideline_ll = die_ll - 1e-4;
    at::Tensor single_sideline_ur = die_ur + die_ll - 1e-4;
    at::Tensor via_mov_node_size;
    at::Tensor via_mov_node_pos;
    at::Tensor via_expand_ratio;
    at::Tensor via_node_to_num_pins;
    at::Tensor via_sidelines_ll;
    at::Tensor via_sidelines_ur;
    int via_mov_lhs = 0;
    int via_mov_rhs = num_nets;
    if (true) {
        at::Tensor die_area = at::prod(die_ur - die_ll);
        at::Tensor via_size_x = (bondingInfo[0] + bondingInfo[2]) * 1;
        at::Tensor via_size_y = (bondingInfo[1] + bondingInfo[2]) * 1;
        at::Tensor via_size_z = die_ur[2];
        at::Tensor single_via_size =
            at::tensor({via_size_x.item<float>(), via_size_y.item<float>(), via_size_z.item<float>()},
                       torch::dtype(mov_node_size.dtype()));
        via_node_size = single_via_size.repeat({num_nets, 1});
        at::Tensor via_mov_cell_area = torch::prod(via_node_size, 1);

        // FIXME: filler size normalization
        at::Tensor via_filler_size_x = bondingInfo[0] + bondingInfo[2];
        at::Tensor via_filler_size_y = bondingInfo[1] + bondingInfo[2];
        at::Tensor via_filler_size_z = die_ur[2];
        at::Tensor via_total_filler_area = max(target_density * die_area - torch::sum(via_mov_cell_area),
                                               at::tensor(0.0, dtype(torch::kFloat).device(torch::kCPU)));

        at::Tensor single_via_filler_size = at::tensor(
            {via_filler_size_x.item<float>(), via_filler_size_y.item<float>(), via_filler_size_z.item<float>()},
            torch::dtype(mov_node_size.dtype()));

        int via_num_fillers =
            torch::round(via_total_filler_area / (via_filler_size_x * via_filler_size_y * via_filler_size_z))
                .item<int>();
        via_node_to_num_pins = torch::ones({num_nets + via_num_fillers, 1}, torch::dtype(torch::kFloat));

        at::Tensor via_filler_size;
        bool via_use_filler = true;
        if (via_num_fillers > 0) {
            via_filler_size = single_via_filler_size.repeat({via_num_fillers, 1});
            logger.info("#Via Fillers: %d [%.2f], Filler size: (%.4e, %.4e, %.4e)",
                        via_num_fillers,
                        target_density,
                        single_via_filler_size[0].item<float>(),
                        single_via_filler_size[1].item<float>(),
                        single_via_filler_size[2].item<float>());
        } else {
            via_use_filler = false;
            logger.info("disable via filler");
        }

        /* cat filler */
        via_mov_node_size = via_node_size;

        at::Tensor scale = (die_ur - die_ll) * 0.001;
        ;  // TODO:
        scale[2] *= 1;
        at::Tensor loc = (die_ur + die_ll) * 0.5;
        via_mov_node_pos = torch::randn_like(via_node_size) * scale + loc;
        // FIXME: rand mov node pos
        if (via_use_filler) {
            // TODO: enable fence
            at::Tensor via_filler_pos = torch::rand({via_num_fillers, 3}, torch::dtype(via_node_size.dtype()));
            at::Tensor scale = die_ur - die_ll;
            at::Tensor shift = die_ll;
            via_filler_pos = via_filler_pos * scale + shift;

            via_mov_node_pos = torch::cat({via_mov_node_pos, via_filler_pos}, 0);
            via_mov_node_size = torch::cat({via_mov_node_size, via_filler_size}, 0);
        }
        if (st::setting.noise_ratio > 0) {
            at::Tensor noise = torch::rand_like(via_mov_node_pos, torch::dtype(via_mov_node_pos.dtype()));
            noise = noise.sub(0.5).mul(via_mov_node_size).mul(st::setting.noise_ratio);
            via_mov_node_pos += noise;
        }

        via_expand_ratio = torch::ones((via_mov_node_pos.size(0)), torch::dtype(via_mov_node_size.dtype()));
        at::Tensor single_sideline_ll = die_ll - 1e-4;
        at::Tensor single_sideline_ur = die_ur + die_ll - 1e-4;

        via_sidelines_ll = single_sideline_ll.repeat({via_mov_node_pos.size(0), 1});
        via_sidelines_ur = single_sideline_ur.repeat({via_mov_node_pos.size(0), 1});
    }

    /* update mov_node info */
    auto mov_node_size_all = torch::cat({mov_node_size.index({Slice({mov_lhs, mov_rhs})}),
                                         via_mov_node_size.index({Slice({via_mov_lhs, via_mov_rhs})}),
                                         mov_node_size.index({Slice({mov_rhs, torch::indexing::None})}),
                                         via_mov_node_size.index({Slice({via_mov_rhs, torch::indexing::None})})},
                                        0);
    auto mov_node_pos_all = torch::cat({mov_node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                        via_mov_node_pos.index({Slice({via_mov_lhs, via_mov_rhs})}),
                                        mov_node_pos.index({Slice({mov_rhs, torch::indexing::None})}),
                                        via_mov_node_pos.index({Slice({via_mov_rhs, torch::indexing::None})})},
                                       0);  // FIXME: which order

    auto expand_ratio_all = torch::cat({expand_ratio.index({Slice({mov_lhs, mov_rhs})}),
                                        via_expand_ratio.index({Slice({via_mov_lhs, via_mov_rhs})}),
                                        expand_ratio.index({Slice({mov_rhs, torch::indexing::None})}),
                                        via_expand_ratio.index({Slice({via_mov_rhs, torch::indexing::None})})},
                                       0);

    sidelines_ll = torch::cat({sidelines_ll.index({Slice({mov_lhs, mov_rhs})}),
                               via_sidelines_ll.index({Slice({via_mov_lhs, via_mov_rhs})}),
                               sidelines_ll.index({Slice({mov_rhs, torch::indexing::None})}),
                               via_sidelines_ll.index({Slice({via_mov_rhs, torch::indexing::None})})},
                              0);

    sidelines_ur = torch::cat({sidelines_ur.index({Slice({mov_lhs, mov_rhs})}),
                               via_sidelines_ur.index({Slice({via_mov_lhs, via_mov_rhs})}),
                               sidelines_ur.index({Slice({mov_rhs, torch::indexing::None})}),
                               via_sidelines_ur.index({Slice({via_mov_rhs, torch::indexing::None})})},
                              0);

    mov_node_to_num_pins = torch::cat({mov_node_to_num_pins.index({Slice({mov_lhs, mov_rhs})}),
                                       via_node_to_num_pins.index({Slice({via_mov_lhs, via_mov_rhs})}),
                                       mov_node_to_num_pins.index({Slice({mov_rhs, torch::indexing::None})}),
                                       via_node_to_num_pins.index({Slice({via_mov_rhs, torch::indexing::None})})},
                                      0);

    auto mov_node_weights_all =
        torch::cat({torch::ones(mov_rhs - mov_lhs, torch::dtype(torch::kFloat)),
                    torch::zeros(via_mov_rhs - via_mov_lhs, torch::dtype(torch::kFloat)),
                    torch::ones(mov_node_size.size(0) - mov_rhs, torch::dtype(torch::kFloat)),
                    torch::zeros(via_mov_node_size.size(0) - via_mov_rhs, torch::dtype(torch::kFloat))},
                   0);

    /* update node_die: -1 | 0 | 1 | 2 */
    node_die = torch::cat(
        {torch::ones({mov_rhs}, torch::dtype(torch::kInt)), -1 * torch::ones({via_mov_rhs}, torch::dtype(torch::kInt))},
        0);
    mov_node_weights[0] = mov_node_weights_all.clone();
    mov_node_weights[1] = (1 - mov_node_weights_all).clone();

    /* update pin_rel_cpos */
    torch::Tensor via_pin_rel_cpos = torch::zeros({num_nets, 3}, torch::dtype(mov_node_pos.dtype()));
    pin_rel_cpos = torch::cat({pin_rel_cpos, via_pin_rel_cpos}, 0);

    /* update node area info | sorted maps */
    movable_index = make_tuple(mov_lhs, mov_rhs + via_mov_rhs);
    std::tie(mov_lhs, mov_rhs) = movable_index;
    node_type_indices.emplace_back(std::make_tuple(cell_mov_rhs, mov_rhs, "Via"));

    /* compute_sorted_node_map */
    // TODO: for parameter_scheduler
    // node_size = mov_node_size_all.index({Slice({mov_lhs, mov_rhs})}).clone();

    node_size = torch::cat({node_size, via_node_size}, 0);  // FIXME: original size before expand
    mov_node_area = torch::prod(mov_node_size_all, 1).unsqueeze(1);
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);
    mov_sorted_map = mov_sorted_map.contiguous();
    auto [tmp1, mov_conn_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
    auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
    mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
    filler_sorted_map = filler_sorted_map.contiguous();
    sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));

    return make_tuple(mov_node_pos_all, mov_node_size_all, expand_ratio_all);
}  // END MODULE

void NodeData3D::update_net_weight(torch::Tensor box_len_at) {
    logger.info("Nets smaller than %.2f are ignored", st::setting.weaken_net_size);

    box_len_at.index({"...", 0}) /= die_info[1].item<float>();
    box_len_at.index({"...", 1}) /= die_info[3].item<float>();
    const torch::TensorAccessor<float, 2> box_len = box_len_at.accessor<float, 2>();

    for (int i = 0; i < num_nets; i++) {
        if (box_len[i][0] < st::setting.weaken_net_size && box_len[i][1] < st::setting.weaken_net_size) {
            net_weight[i] *= 0.01 * net_to_num_pins[i];
        } else {
            net_weight[i] *= 10;
        }
    }
}