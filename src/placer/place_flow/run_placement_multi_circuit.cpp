#include "../run_placement.h"
#include "fp/fp/floorplan.h"
#include "utils/myUtils.h"

void run_placement_main_multi_circuit() {
    /* General settings */
    logger.info("#threads %d", st::setting.num_threads);
    torch::set_num_threads(st::setting.num_threads);
    setenv("OMP_NUM_THREADS", to_string(st::setting.num_threads).c_str(), true);

    /* set device */
    torch::Device device = torch::kCPU;
    int num_device = torch::cuda::device_count();
    logger.info("CUDA DEVICE COUNT: %d", num_device);
    if (torch::cuda::is_available() && (st::setting.gpu < num_device) && (st::setting.gpu >= 0)) {
        logger.info("CUDA is available! Training on GPU %d.", st::setting.gpu);
        // device = torch::kCUDA;

        device = torch::Device(torch::kCUDA, st::setting.gpu);
    }

    logger.info("Use Nesterov optimizer!");
    if (st::setting.scale_design) {
        logger.warning("Eplace's nesterov optimizer cannot support normalized die. Disable scale_design.");
        st::setting.scale_design = false;
    }

    /* tmp output for verification */
    std::string output_path_tmp = st::setting.output_path + ".tmp";

    /* visualization via color */
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    // ======================================================================================================
    //
    //                                             PARTITION
    //
    // ======================================================================================================
    /* database */
    auto [design_info, rawdb, gpdb] = load_dataset();
    NodeData data(design_info, device);
    auto macro_mask_2d = data.macro_mask.clone().unsqueeze(1);
    // data.setMacroOrient_vertical();
    // grad.slice(0, 0, macro_mask_2d.size(0)) *= (1-0.99*macro_mask_2d);
    // for(int i=0;i<data.pin_rel_cpos.size(0);i++)
    // {
    //     int node_id = data.pin_id2node_id[i].item<int>();
    //     if(data.macro_mask[node_id].item<int>()==1)
    //     {
    //         data.pin_rel_cpos_bot[i]*=(data.node_size_bot[0]/data.node_size_bot[node_id]);
    //         data.pin_rel_cpos_top[i]*=(data.node_size_top[0]/data.node_size_top[node_id]);
    //         data.pin_rel_cpos[i]*=(data.node_size[0]/data.node_size[node_id]);
    //     }
    // }

    // for(int i=0;i<data.node_size_bot.size(0);i++)
    // {
    //     if(data.macro_mask[i].item<int>()==1)
    //     {
    //         data.node_size_bot[i]=data.node_size_bot[0];
    //         data.node_size_top[i]=data.node_size_top[0];
    //         data.node_size[i]=data.node_size[0];
    //         data.node_area_bot[i]=data.node_size[i][0]*data.node_size[i][1];

    //     }
    // }
    data.preprocess();
    torch::Tensor node_die;
    torch::Tensor node_pos;
    torch::Tensor node_size;

    states hpwl_state;

    /* node information */
    int cell_mov_lhs;
    int cell_mov_rhs;
    int via_mov_lhs;
    int via_mov_rhs;
    int mov_lhs;
    int mov_rhs;
    torch::Tensor node_size_bot;
    torch::Tensor node_size_top;
    torch::Tensor node_orient_bot;
    torch::Tensor node_orient_top;
    /* cell visualization information */
    node_size_bot = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    node_size_top = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    std::tie(cell_mov_lhs, cell_mov_rhs) = data.movable_index;
    logger.info("#mode: %d", st::setting.mode);

    /* pt/via database */
    ViaData via_data;


    /* update data info function */
    std::function<void(torch::Tensor)> update_model_fn =
        [&data, &via_data, &rawdb, &node_size_bot, &node_size_top](torch::Tensor node_die) {
            // ViaData via_data(data, rawdb, node_die);        // FIXME: need polish
            via_data = ViaData(data, rawdb, node_die);
            data.node_die = node_die.clone();
            data.mov_cell_areas = via_data.mov_cell_areas;
            auto init_density_map_tmp = get_init_density_map(data);

            auto [mov_node_pos_tmp, mov_node_size_tmp, expand_ratio_tmp] = data.get_mov_node_info_cross_chip();

            via_data.init_vars();  // TODO: equivalent to PlaceData::init_filler
            auto [via_mov_node_pos_tmp, via_mov_node_size_tmp, via_expand_ratio_tmp] = via_data.get_mov_node_info();

            /* movable cells&vias*/
            auto [tmp1, tmp2, tmp3] = data.get_mov_node_info_cross_chip_with_via(
                via_data,
                make_tuple(mov_node_pos_tmp, mov_node_size_tmp, expand_ratio_tmp),
                make_tuple(via_mov_node_pos_tmp, via_mov_node_size_tmp, via_expand_ratio_tmp));

            node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
            node_size_top = data.node_size_top * node_die.unsqueeze(1);
        };

    torch::Tensor partial_hpwl3d;
    Partitioner pt(data, hpwl_state);
    /* Partitioning */
    if (st::setting.pt) {
        if (st::setting.partitioner == "fm") {
            pt.run();
        } else if (st::setting.partitioner == "gp2d_grid") {
            node_pos = pt.run_gp2d_grid(data);
            st::setting.use_filler = true;
        } else if (st::setting.partitioner == "patoh") {
            pt.run_patoh(data);
        } else if (st::setting.partitioner == "gp3d") {
            // if (!st::setting.force_coeff_2d) {
            if (true) {
                // auto data_size_backup0 = data.node_size.clone();
                // auto macro_mask_2d = data.macro_mask.unsqueeze(1);
                // data.node_size.slice(0, 0, macro_mask_2d.size(0))*=(1-0.99*macro_mask_2d);//@@FREEZE
                if(st::setting.mode==4)
                {
                    st::setting.magic_hpwl=350000;
                }
                if(st::setting.load_stage=="2D")
                {
                    data.node_die = torch::ones(data.num_nodes, torch::dtype(torch::kInt));
                    node_pos = load_from_file(st::setting.load_file, data);
                    st::setting.cache_macro_mask = data.macro_mask.to(data.device);
                }
                else{
                    node_pos = pt.run_gp2d_grid(data);
                    // pt.run_patoh_grided(data, node_pos);
                    data.node_die = torch::ones(data.num_nodes, torch::dtype(torch::kInt));
                    // via_data = ViaData(data, rawdb, data.node_die);
                    // auto node_size_2d = data.node_size;
                    // std::filesystem::path current_dir(std::filesystem::current_path());
                    // std::filesystem::path result_dir(st::setting.result_dir);
                    // std::filesystem::path exp_id(st::setting.exp_id);
                    // std::filesystem::path filename("gp2d_output.txt");
                    // std::filesystem::path file_path = (current_dir / result_dir / exp_id / filename);
                    // auto via_pos = torch::ones({data.num_nets,2});
                    // auto via_die = torch::ones({data.num_nets});
                    // via_die *= -1;
                    // via_data.dump(torch::cat({node_pos.slice(1,0,2), via_pos}, 0), node_size_2d,torch::cat({data.node_die,via_die}), cell_mov_lhs, cell_mov_rhs, data.node_orient_top);
                    // rawdb->writeICCAD2022(file_path);
                }
                // data.node_size = data_size_backup0;
                pt.node_pos_2d_ground = node_pos;
                // data.node_pos = node_pos;
            }
            if (st::setting.force_coeff_2d == 1) st::setting.use_filler_3d = true;
            // pt.run_patoh(data);
            // pt.greedy_macro_partition(data);//@@FREEZE
            // pt.greedy_partition_by_area(data);
            // if(st::setting.mode==3)
            // {
            //     pt.run_patoh(data);
            // }else{
            //     pt.greedy_partition_by_std_area(data);
            // }
            // pt.greedy_partition_by_std_area(data);
            // pt.greedy_partition_by_maximize_cuts(data);
            data.node_die = pt.node_die.clone();
            node_die = pt.node_die.clone();
            // std::tie(sorted_tensor, indices) = tensor.sort();
            if(st::setting.adjust_macro)
            {
                pt.adjust_macros(data);
                node_die = pt.node_die.clone();
            }
            // data.node_die = pt.node_die.clone();
            if (false) {
                torch::Tensor node_size_top =
                    data.node_size_top * pt.node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
                torch::Tensor node_size_bot =
                    data.node_size_bot * (1 - pt.node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
                auto cell_node_pos_lg = data.node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)});
                auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_BEFORE_GP3D_0");
                draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info1);
                auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_BEFORE_GP3D_1");
                draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info2);
                auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_BEFORE_GP3D_2");
                auto node_pos_draw_cp = torch::cat({cell_node_pos_lg, cell_node_pos_lg}, 0);
                auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
                draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
            }
            data.node_die = pt.node_die.clone();
            if(st::setting.mode==4)
            {
                st::setting.magic_hpwl=350000;
            }
            if(st::setting.load_stage=="3D") {
                data.node_die = torch::ones(data.num_nodes, torch::dtype(torch::kInt));
                node_pos = load_from_file(st::setting.load_file, data);
            }
            else {
                auto node_die_check = data.node_die.clone();
                if (st::setting.patoh_guide_ratio > 1e-3) {
                    pt.run_patoh_area(data);
                }
                else {
                    data.node_die = torch::rand({data.cell_mov_rhs - data.cell_mov_lhs}).round().to(torch::kInt);
                }
                auto rotate_90 = st::setting.rotate_90 && st::setting.rotate_180;
                auto node_orient_back_up = data.node_orient_top.clone();
                auto stop_overflow_3d_back_up = st::setting.stop_overflow_3d;
                // if (rotate_90) {
                //     st::setting.stop_overflow_3d = 0.2;
                // }
                torch::Tensor macro_indices = torch::nonzero(data.macro_mask).squeeze();
                auto [node_rotate, new_node_pos, node_rotate90_tend] = pt.run_gp3d(data, rotate_90);  // second gp in gp3d mode
                auto node_angle = (node_rotate + data.node_orient_top) % 4;
                data.setMacroOrient(node_angle);
                // if (rotate_90) {
                //     rotate_90 = false;
                //     st::setting.stop_overflow_3d = stop_overflow_3d_back_up;
                //     std::tie(node_rotate, new_node_pos, node_rotate90_tend) = pt.run_gp3d(data, rotate_90);
                //     node_angle = (node_rotate + data.node_orient_top) % 4;
                //     data.setMacroOrient(node_angle);
                // }
                node_pos = new_node_pos.clone();
                auto node_die_check2 = pt.node_die.clone();
                auto node_die_diff = node_die_check.slice(0,data.cell_mov_lhs, data.cell_mov_rhs)^node_die_check2.slice(0,data.cell_mov_lhs, data.cell_mov_rhs);
                int diff_num = node_die_diff.sum().item<int>();
                logger.info("%d nodes changed their die id, total %d, ratio %f", diff_num, data.cell_mov_rhs, float(diff_num)/float(data.cell_mov_rhs));
            }
            data.node_pos = node_pos;

            // via_data = ViaData(data, rawdb, data.node_die);
            // auto node_size555 = data.node_size_bot * (1 - node_die).unsqueeze(1) + data.node_size_top * node_die.unsqueeze(1);
            // std::filesystem::path current_dir(std::filesystem::current_path());
            // std::filesystem::path result_dir(st::setting.result_dir);
            // std::filesystem::path exp_id(st::setting.exp_id);
            // std::filesystem::path filename("gp3d_output.txt");
            // std::filesystem::path file_path = (current_dir / result_dir / exp_id / filename);
            // auto via_pos = torch::ones({data.num_nets,2});
            // auto via_die = torch::ones({data.num_nets});
            // via_die *= -1;
            // via_data.dump(torch::cat({node_pos.slice(1,0,2), via_pos}, 0), node_size555,torch::cat({data.node_die,via_die}), cell_mov_lhs, cell_mov_rhs, data.node_orient_top);
            // rawdb->writeICCAD2022(file_path);
            // if (st::setting.force_coeff_2d) {  // false
            data.node_die = pt.node_die.clone();
        } else {
            printlog(LOG_ERROR, "Partitioner %s not found!", st::setting.partitioner.c_str());
            pt.node_die = torch::ones(data.num_nodes, dtype(torch::kInt));
        }

        node_die = pt.node_die.clone();
        data.node_die = pt.node_die.clone();
        data.mov_cell_areas = pt.mov_cell_areas.clone();

        if (true) {
            int count = 0;
            for (int i = 0; i < data.num_nodes; i++)
                if (data.aspect_ratio[i].item<float>() > 6)
                    if (node_die[i].item<int>() == 1) count++;
            //         if (node_die[i].item<int>() != st::setting.stack_cells) count++;
            // cout << "=========== " << count << " cells not placed right" << endl;
            logger.info("[%d / %d] long cells at chip [0/1]",
                        torch::_cast_Float(data.aspect_ratio > 6).sum().item<int>() - count,
                        count);
        }

        if (node_pos.numel()) {
            /* HPWL metrics */
            auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), node_die.to(device), data);
            data.hyperedge_list = data.hyperedge_list.to(device);
            data.hyperedge_list_end = data.hyperedge_list_end.to(device);
            torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
                node_pos.to(device), data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device));
            torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach()));
            float hpwl_float = (hpwl + hpwl_ovlp).item<float>();
            // logger.info("======= Before GP, estimated wl cross-chip [%d + %d = %d] =======", hpwl.item<int>(),
            //             hpwl_ovlp.item<int>(), (hpwl + hpwl_ovlp).item<int>());
            logger.info("======= Before GP, estimated wl cross-chip [%f + %f = %f] =======",
                        hpwl.item<float>(),
                        hpwl_ovlp.item<float>(),
                        (hpwl + hpwl_ovlp).item<float>());
        }

        if (node_pos.numel()) {
            node_pos = node_pos.index({"...", Slice(0, 2)}).to(device);
            data.node_pos = node_pos;
            data.to(torch::kCPU);
        }

        st::setting.magic_hpwl = st::setting.__ori_magic_hpwl__;
        st::setting.density_weight = st::setting.__ori_density_weight__;
        st::setting.density_weight_coef = st::setting.__ori_density_weight_coef__;
        st::setting.wa_coeff = st::setting.__ori_wa_coeff__;
        st::setting.quad_penalty = st::setting.__ori_quad_penalty__;
    } else if (st::setting.gp_model != "null") {
        logger.info("Loading model from %s", st::setting.gp_model.c_str());
        torch::load(node_pos, st::setting.gp_model);

        string model_dir = st::setting.gp_model;
        std::size_t pos = model_dir.find_last_of("/");
        std::string tmp = model_dir.substr(0, pos + 1);
        string pt_dir = tmp + "pt3d.pt";
        logger.info("Loading PT from %s", pt_dir.c_str());
        torch::load(node_die, pt_dir);

        // node_die = node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        // node_pos = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        data.node_die = node_die.clone();
    }

    node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
    node_size_top = data.node_size_top * node_die.unsqueeze(1);
    // ======================================================================================================
    //
    //                                             MACRO FLOORPLAN
    //
    // ======================================================================================================
    if (st::setting.mode == 1) {
        // for (int i = 0; i < data.cell_mov_rhs; i++) {
        //     data.node_pos[i] = 0;
        // }
        // data.node_die[0] = 0;
        // data.node_die[1] = 0;
        // data.node_die[3] = 0;
        // data.node_die[4] = 0;
        // data.node_die[2] = 1;
        // data.node_die[5] = 1;
        // data.node_die[6] = 1;
        // data.node_die[7] = 1;
        run_greedy_place(data);
        torch::Tensor node_size_top =
            data.node_size_top * data.node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
        torch::Tensor node_size_bot =
            data.node_size_bot * (1 - data.node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
        auto cell_node_pos_lg = data.node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_CASE1_CHIP_0");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_CASE1_CHIP_1");
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_CASE1_CHIP_2");
        auto node_pos_draw_cp = torch::cat({cell_node_pos_lg, cell_node_pos_lg}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp(node_pos_draw_cp, node_size_draw_cp, data, info3);
        via_data = ViaData(data, rawdb, data.node_die);
        node_pos = data.node_pos;
        node_size = data.node_size_top * data.node_die.unsqueeze(1) + data.node_size_bot * (1 - data.node_die).unsqueeze(1);

        // torch::Tensor init_density_map = get_init_density_map(data);
        // auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info_cross_chip();

        // auto [via_mov_node_pos, via_mov_node_size, via_expand_ratio] = via_data.get_mov_node_info();
        // auto [mov_node_pos_all, mov_node_size_all, expand_ratio_all] = data.get_mov_node_info_cross_chip_with_via(
        //     via_data, make_tuple(mov_node_pos, mov_node_size, expand_ratio),
        //     make_tuple(via_mov_node_pos, via_mov_node_size, via_expand_ratio));
        torch::Tensor init_density_maps = get_init_density_map_cross_chip(data);
        data.init_density_maps = data.init_density_maps.cpu();
        
        update_model_fn(data.node_die);
        mov_rhs = cell_mov_rhs + data.num_nets;
        node_pos = data.node_pos;

        auto via_pos = torch::zeros({data.num_nets, 2});
        node_pos = torch::cat({node_pos, via_pos}, 0);



        // via_data = ViaData(data, rawdb, node_die);
        // data.node_die = node_die.clone();
        // data.mov_cell_areas = via_data.mov_cell_areas;
        // auto init_density_map_tmp = get_init_density_map(data);
        // auto [mov_node_pos_tmp, mov_node_size_tmp, expand_ratio_tmp] = data.get_mov_node_info_cross_chip(false,false);
        // via_data.init_vars();  // TODO: equivalent to PlaceData::init_filler
        // auto [via_mov_node_pos_tmp, via_mov_node_size_tmp, via_expand_ratio_tmp] = via_data.get_mov_node_info();
        // /* movable cells&vias*/
        // auto [move_node_pos_all, tmp2, tmp3] = data.get_mov_node_info_cross_chip_with_via(
        //     via_data,
        //     make_tuple(mov_node_pos_tmp, mov_node_size_tmp, expand_ratio_tmp),
        //     make_tuple(via_mov_node_pos_tmp, via_mov_node_size_tmp, via_expand_ratio_tmp));
    }

    if (st::setting.use_floorplan) {
        logger.info("Start Floorplanning...\n");
        torch::Tensor init_density_maps = get_init_density_map_cross_chip(data);
        data.init_density_maps = data.init_density_maps.cpu();
        node_pos = node_pos.to(torch::kCPU);

        data.get_mov_node_info_cross_chip();
        torch::Tensor node_pos1 = run_floorplan(data, node_pos, hpwl_state, cell_mov_lhs, cell_mov_rhs);
        node_pos = node_pos1.clone();
        logger.info("finish running fp");

        data.node_pos = node_pos;

        node_pos = node_pos.to(device);
        node_die = data.node_die.clone();
    }
    // ======================================================================================================
    //
    //                                             GP CELL && VIA
    //
    // ======================================================================================================
    if (st::setting.gp) {
        /* movable vias */
        logger.info("getting viadata");
        int magic_backup = st::setting.magic_hpwl;
        // int num_bin_x_backup = st::setting.num_bin_x;
        // int num_bin_y_backup = st::setting.num_bin_y;
        if (!st::setting.use_floorplan) {
            via_data = ViaData(data, rawdb, node_die);
            logger.info("finish getting viadata");
            auto node_die_backup = data.node_die.clone();
            auto node_pos_backup = node_pos.clone();

            int is_move_macro = true;
            int is_init_macro = true;
            int is_init_stdcell = true;
            if (st::setting.use_floorplan) {
                is_move_macro = false;
                is_init_macro = false;
            }
            if(st::setting.mode==4)
            {
                st::setting.magic_hpwl=70000;
            }
            st::setting.magic_hpwl=70000;
            node_pos = run_gp(data,
                              via_data,
                              node_pos,
                              hpwl_state,
                              cell_mov_lhs,
                              cell_mov_rhs,
                              via_mov_lhs,
                              via_mov_rhs,
                              mov_lhs,
                              mov_rhs,
                              is_move_macro,
                              is_init_macro,
                              is_init_stdcell,
                              "first");
                            //   "first"+to_string(ii));  // third gp for gp3d mode
            std::filesystem::path current_dir(std::filesystem::current_path());
            std::filesystem::path result_dir(st::setting.result_dir);
            std::filesystem::path exp_id(st::setting.exp_id);
            std::filesystem::path filename("gp2.5d_1_output.txt");
            std::filesystem::path file_path = (current_dir / result_dir / exp_id / filename);
            auto data_size_check = data.node_size_bot * (1 - node_die).unsqueeze(1) + data.node_size_top * node_die.unsqueeze(1);
            via_data.dump(node_pos, data_size_check, data.node_die, cell_mov_lhs, cell_mov_rhs, data.node_orient_top);
            rawdb->writeICCAD2022(file_path);
            node_size = data.node_size.clone();
        }
        if(st::setting.mode==4)
        {
            // st::setting.magic_hpwl = magic_backup;
            st::setting.magic_hpwl = 70000;
            // st::setting.num_bin_x = st::setting.second_num_bin_x;
            // st::setting.num_bin_y = st::setting.second_num_bin_y;
            // data.num_bin_x = st::setting.second_num_bin_x;
            // data.num_bin_y = st::setting.second_num_bin_y;
            //st::setting.stop_overflow = st::setting.stop_overflow / 5;
        }
        

        for (; st::setting.round_recursion >= 0;) {
            /* hpwl-driven fm */
            if (true) {
                // node_pos = pt.run_gp3d(data);
                logger.info("============== HPWL-FM round %d ==============", st::setting.round_recursion);
                data.reset();
                pt.node_die = node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).clone();
                data.node_die = data.node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)});
                pt.node_pos_2d_ground = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs), Slice(0, 2)});
                data.node_pos = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs), Slice(0, 2)});
                data.node_size =
                    data.node_size_bot * (1 - pt.node_die).unsqueeze(1) + data.node_size_top * pt.node_die.unsqueeze(1);

                pt.run_fm_wl(data);
            }

            node_die = pt.node_die.clone();
            data.node_die = pt.node_die.clone();
            data.mov_cell_areas = pt.mov_cell_areas.clone();
            node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
            node_size_top = data.node_size_top * node_die.unsqueeze(1);  //@@@@@

            st::setting.num_den_layer = (st::setting.round_recursion == 0) ? 3 : 2;
            via_data = ViaData(data, rawdb, node_die);
            node_pos = run_gp(data,
                              via_data,
                              node_pos,
                              hpwl_state,
                              cell_mov_lhs,
                              cell_mov_rhs,
                              via_mov_lhs,
                              via_mov_rhs,
                              mov_lhs,
                              mov_rhs,
                              true,
                              true,
                              true,
                              "first");
        }

        if (st::setting.rf) {
            node_pos = run_gp_refine(data,
                                     via_data,
                                     node_pos,
                                     hpwl_state,
                                     cell_mov_lhs,
                                     cell_mov_rhs,
                                     via_mov_lhs,
                                     via_mov_rhs,
                                     mov_lhs,
                                     mov_rhs);
            mov_rhs = cell_mov_rhs + data.num_nets;
            data.fixed_connected_index = make_tuple(mov_lhs, mov_rhs);
        }
    } else if (st::setting.gp_model != "null") {
        logger.info("Loading model from %s", st::setting.gp_model.c_str());
        torch::load(node_pos, st::setting.gp_model);

        string model_dir = st::setting.gp_model;
        std::size_t pos = model_dir.find_last_of("/");
        std::string tmp = model_dir.substr(0, pos + 1);
        string pt_dir = tmp + "pt.pt";
        logger.info("Loading PT from %s", pt_dir.c_str());
        torch::load(node_die, pt_dir);

        node_die = node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        // node_pos = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        via_data = ViaData(data, rawdb, node_die);
        data.node_die = node_die.clone();
        data.mov_cell_areas = via_data.mov_cell_areas;
    }

    // data.reset();
    via_data.postscale();
    node_pos *= data.site_width;
    node_size *= data.site_width;
    data.postscale_by_site_width();
    node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
    node_size_top = data.node_size_top * node_die.unsqueeze(1);

    /* Legalization */
    if (st::setting.lg) {
        node_pos = run_lg(data,
                          via_data,
                          node_pos,
                          hpwl_state,
                          cell_mov_lhs,
                          cell_mov_rhs,
                          via_mov_lhs,
                          via_mov_rhs,
                          mov_lhs,
                          mov_rhs,
                          false);
    } else if (st::setting.lg_model != "null") {  // TODO: add to argument
        logger.info("Lading model from %s", st::setting.lg_model.c_str());
        torch::load(node_pos, st::setting.lg_model);

        string model_dir = st::setting.lg_model;
        std::size_t pos = model_dir.find_last_of("/");
        std::string tmp = model_dir.substr(0, pos + 1);
        string pt_dir = tmp + "pt.pt";
        logger.info("Loading PT from %s", pt_dir.c_str());
        torch::load(node_die, pt_dir);

        update_model_fn(node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}));

        std::tie(mov_lhs, mov_rhs) = data.movable_index;
        auto [hpwl1, hpwl2, tmp1] = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
        hpwl_state.hpwls_lg = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];
    }

    // via_data.dump(node_pos, node_size, data.node_die, cell_mov_lhs, cell_mov_rhs, data.node_orient_top);
    // std::filesystem::path current_dir(std::filesystem::current_path());
    // std::filesystem::path result_dir(st::setting.result_dir);
    // std::filesystem::path exp_id(st::setting.exp_id);
    // std::filesystem::path filename("gp2.5d_2_lg_output.txt");
    // std::filesystem::path file_path = (current_dir / result_dir / exp_id / filename);
    // // rawdb->writeICCAD2022("/data/ssd/lxiao23/6_2/xyzplace/lg_output.txt");
    // rawdb->writeICCAD2022(file_path);
    // rawdb->writeICCAD2022("/data/ssd/lxiao23/6_2/xyzplace/lg_output.txt");

    // /* post process */  // TODO:
    // if (st::setting.pp) {
    //     /* fetch data info */
    //     torch::Tensor via_node_pos_init = node_pos.index({Slice(cell_mov_rhs, mov_rhs)});
    //     torch::Tensor via_node_pos_pp = via_node_pos_init.clone();
    //     torch::Tensor via_node_size_pp = via_data.node_size;
    //     torch::Tensor via_node_weight = via_data.bonding_map.clone();
    //     post_process(data, node_pos, data.node_die, via_node_pos_pp, via_node_size_pp, via_data.die_info[2],
    //                  via_data.numRows, via_data.row_height);
    //     auto node_pos_pp = torch::cat({node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_node_pos_pp}, 0);

    //     auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_pp.to(device), data.node_die.to(device), data);
    //     logger.info("");
    //     logger.info("After Post Process, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<float>(),
    //                 hpwl2.item<float>(), (hpwl1 + hpwl2).item<float>());
    //     hpwls[hpwl_idx++] = (hpwl1 + hpwl2).item<float>();
    //     logger.info("hpwl %.3f => %.3f (imp. %g%%)", hpwls[hpwl_idx - 2], hpwls[hpwl_idx - 1],
    //                 (1.0 - hpwls[hpwl_idx - 1] / (double)hpwls[hpwl_idx - 2]) * 100);

    //     node_pos = node_pos_pp;

    //     printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(),
    //              utils::mem_use::get_peak());
    // }

    /* Detailed placement */
    if (st::setting.dp) {
        // if (false) {
        /* fetch data info */
        torch::Tensor node_pos_dp = node_pos.clone();
        torch::Tensor node_size_dp = data.node_size.clone();
        // node_size_dp.index({Slice(cell_mov_rhs, None)}) -= data.bondingInfo[2];  // FIXME: w+space -> w
        dp::DetailedPlaceDataTensor dp_db_at(data, node_pos_dp, node_size_dp);

        auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos_dp.to(device), data.node_die.to(device), data);
        logger.info("");
        auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
            evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        /*
        logger.notice("Init DP, exact HPWL [bot + top = total / overlap / overhead]: [%d + %d = %d / %d / %d]",
                      hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
                      hpwl_ovhd.item<int>());
        */
        logger.notice(
            "Init DP, exact HPWL [bot + top = total / overlap / overhead]: [%.2f + %.2f = %.2f / %.2f / %.2f]",
            hpwl1.item<float>(),
            hpwl2.item<float>(),
            (hpwl1 + hpwl2).item<float>(),
            hpwl_ovlp.item<float>(),
            hpwl_ovhd.item<float>());

        for (int i = 0; i < 2; i++) {
            logger.info("======================= Chip-%d-DP =======================", i);
            /* update node_weight for each chip */
            torch::Tensor node_weight = data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)});
            torch::Tensor via_node_weight = via_data.bonding_map.clone();
            node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));

            dp_db_at.update_node_weight(node_weight);
            dp::detail_placement(data,
                                 dp_db_at,
                                 node_pos_dp,
                                 data.numRows[i].item<int>(),
                                 data.rowHeights[i].item<float>(),
                                 data.num_bin_x,
                                 data.num_bin_y);
        }
        if (st::setting.via_dp) {
            logger.info("======================= Via-DP =======================");
            torch::Tensor node_weight = torch::zeros({cell_mov_rhs}, (data.mov_node_weights[0].dtype()));
            torch::Tensor via_node_weight = via_data.bonding_map.clone();
            node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));

            torch::Tensor node_die1 = torch::ones({cell_mov_rhs}, (data.mov_node_weights[0].dtype()));
            via_node_weight = via_data.bonding_map.clone();
            // node_die1 = torch::_cast_Int(torch::cat({node_die1, via_node_weight}, 0));
            node_die1 = data.node_die.clone();

            dp_db_at.update_node_weight(node_weight, node_die1);
            dp::DetailedPlaceData dp_db(
                data, dp_db_at, via_data.numRows.item<int>(), via_data.row_height.item<float>());
            bool via_dp = true;
            dp_db.xl = via_data.core_info[0].item<float>();
            dp_db.xh = via_data.core_info[1].item<float>();
            dp_db.yl = via_data.core_info[2].item<float>();
            dp_db.yh = via_data.core_info[3].item<float>();
            dp_db.num_movable_nodes = data.num_nodes;

            dp_db.via_dp = via_dp;
            dp_db.i_bgn = via_dp ? dp_db.num_movable_nodes : 0;
            dp_db.i_end = via_dp ? dp_db.num_nodes : dp_db.num_movable_nodes;

            dp::kReorder(dp_db, data.num_bin_x, data.num_bin_y);
            dp_db_at.update_node_pos(node_pos_dp);
            std::tie(hpwl1, hpwl2, hpwl_ovlp) =
                evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
            /*
            logger.info("After 1st K-Reorder, solution eval, exact HPWL [bot + top = total]: [%d + %d = %d]",
                        hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
            */
            logger.info("After 1st K-Reorder, solution eval, exact HPWL [bot + top = total]: [%.2f + %.2f= %.2f]",
                        hpwl1.item<float>(),
                        hpwl2.item<float>(),
                        (hpwl1 + hpwl2).item<float>());
            int via_cost = calc_via_cost(node_die, data) * data.bondingCost;
            logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);
            // independentSetMatching(dp_db, data.num_bin_x, data.num_bin_y);
            // independentSetMatching(dp_db, data.num_bin_x, data.num_bin_y, 2048, 256, 100);
            independentSetMatching(dp_db, data.num_bin_x, data.num_bin_y, 2048, 256, 15);
            dp_db_at.update_node_pos(node_pos_dp);
            std::tie(hpwl1, hpwl2, hpwl_ovlp) =
                evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
            /*
            logger.info("After Independent Set Matching, solution eval, exact HPWL [bot + top = total]: [%d + %d = %d]",
                        hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
            */
            logger.info(
                "After Independent Set Matching, solution eval, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                hpwl1.item<float>(),
                hpwl2.item<float>(),
                (hpwl1 + hpwl2).item<float>());
            /*
            dp::globalSwap(dp_db, data.num_bin_x / 2, data.num_bin_y / 2, 2, 32);
            dp_db_at.update_node_pos(node_pos_dp);
            std::tie(hpwl1, hpwl2, hpwl_ovlp) =
                evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
            logger.info("After Global Swap, solution eval, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                        hpwl1.item<float>(),
                        hpwl2.item<float>(),
                        (hpwl1 + hpwl2).item<float>());
            */
            

            /*
            dp::kReorder(dp_db, data.num_bin_x, data.num_bin_y);
            dp_db_at.update_node_pos(node_pos_dp);
            std::tie(hpwl1, hpwl2, hpwl_ovlp) =
                evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
            logger.info("After 2nd K-Reorder, solution eval, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                        hpwl1.item<float>(),
                        hpwl2.item<float>(),
                        (hpwl1 + hpwl2).item<float>());
            */
        }

        /* draw cells */
        if (true) {
            auto true_node_pos_dp = node_pos_dp.index({Slice(cell_mov_lhs, cell_mov_rhs)});
            auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos_dp);
            true_node_pos_dp = true_node_pos_dp + node_shift;
            auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_DP_0");
            draw_fig_with_cairo_cpp(true_node_pos_dp, node_size_bot, data, info1);
            auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_DP_1");
            draw_fig_with_cairo_cpp(true_node_pos_dp, node_size_top, data, info2);
        }

        node_pos = node_pos_dp;
        /* save to .pt model */
        if (st::setting.save_model) {
            std::filesystem::path current_dir(std::filesystem::current_path());
            std::filesystem::path result_dir(st::setting.result_dir);
            std::filesystem::path exp_id(st::setting.exp_id);
            std::filesystem::path res_root = current_dir / result_dir / exp_id;
            std::string model_dir = res_root.string() + "/dp.pt";
            logger.info("Save model to %s", model_dir.c_str());
            torch::save(node_pos, model_dir);
        }
        std::tie(hpwl1, hpwl2, hpwl_ovlp) = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        std::tie(tmp1, tmp2, hpwl_ovhd, via_bbox) =
            evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        /*
        logger.notice("After DP, exact HPWL [bot + top = total / overlap / overhead]: [%d + %d = %d / %d / %d]",
                      hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
                      hpwl_ovhd.item<int>());
        */
        logger.notice(
            "After DP, exact HPWL [bot + top = total / overlap / overhead]: [%.2f + %.2f = %.2f / %.2f / %.2f]",
            hpwl1.item<float>(),
            hpwl2.item<float>(),
            (hpwl1 + hpwl2).item<float>(),
            hpwl_ovlp.item<float>(),
            hpwl_ovhd.item<float>());
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
        hpwl_state.hpwls_dp = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];
        hpwl_state.log();
        printlog(
            LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    }

    
    // node_pos *= data.site_width;
    // node_size *= data.site_width;
    // data.postscale_by_site_width();
    // node_size_bot = data.node_size_bot * (1 - node_die).unsqueeze(1);
    // node_size_top = data.node_size_top * node_die.unsqueeze(1);

    /* Detailed placement round 2 */
    // if (st::setting.dp) {
    if (false) {
        /* fetch data info */
        torch::Tensor node_pos_dp = node_pos.clone();
        torch::Tensor node_size_dp = data.node_size.clone();
        // node_size_dp.index({Slice(cell_mov_rhs, None)}) -= data.bondingInfo[2];  // FIXME: w+space -> w
        dp::DetailedPlaceDataTensor dp_db_at(data, node_pos_dp, node_size_dp);

        auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_dp.to(device), data.node_die.to(device), data);
        logger.info("");
        logger.info("After DP, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        int via_cost = calc_via_cost(node_die, data) * data.bondingCost;
        logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);
        for (int i = 0; i < 2; i++) {
            logger.info("======================= Chip-%d-DP =======================", i);
            /* update node_weight for each chip */
            torch::Tensor node_weight = data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)});
            torch::Tensor via_node_weight = via_data.bonding_map.clone();
            node_weight = torch::_cast_Int(torch::cat({node_weight, via_node_weight}, 0));

            dp_db_at.update_node_weight(node_weight);
            dp::detail_placement(data,
                                 dp_db_at,
                                 node_pos_dp,
                                 data.numRows[i].item<int>(),
                                 data.rowHeights[i].item<float>(),
                                 data.num_bin_x,
                                 data.num_bin_y);
        }

        node_pos = node_pos_dp;

        torch::Tensor hpwl_ovlp;
        std::tie(hpwl1, hpwl2, hpwl_ovlp) = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        /*
        logger.info("After DP, solution eval, exact HPWL [bot + top = total/extra = 2.5D]: [%d + %d = %d/%d = %d]",
                    hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
                    (hpwl1 + hpwl2 - hpwl_ovlp).item<int>());
        */
        logger.info(
            "After DP, solution eval, exact HPWL [bot + top = total/extra = 2.5D]: [%.2f + %.2f = %.2f/%.2f = %.2f]",
            hpwl1.item<float>(),
            hpwl2.item<float>(),
            (hpwl1 + hpwl2).item<float>(),
            hpwl_ovlp.item<float>(),
            (hpwl1 + hpwl2 - hpwl_ovlp).item<float>());
        // via_cost = calc_via_cost(node_die, data) * data.bondingCost;
        // logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = (hpwl1 + hpwl2).item<float>();
        hpwl_state.log();
        printlog(
            LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    }

    if (st::setting.sw) {
        /* fetch data info */
        logger.info("setting.sw");
        torch::Tensor node_pos_sw = node_pos.clone();
        torch::Tensor node_size_sw = data.node_size.clone();
        node_size_sw.index({Slice(cell_mov_rhs, None)}) *= 0;  // FIXME: zero size: pin_rel_c = pin_rel_l
        dpc::DetailedPlaceDataTensor dp_db_at(data, via_data, node_pos_sw, node_size_sw);
        dpc::DetailedPlaceData dp_db(data, dp_db_at, via_data);
        dpc::globalSwapCrossChip(
            dp_db, data.node_naive_flag, data.numRows, data.rowHeights, data.num_bin_x / 4, data.num_bin_y / 4, 20);
        dp_db_at.update_node_info(data, node_pos_sw, node_size_sw, data.node_die);
        auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_sw.to(device), data.node_die.to(device), data);
        logger.info("");
        /*
        logger.info("After Post Process, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
        */
        logger.info("After Post Process, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        int via_cost = calc_via_cost(node_die, data) * data.bondingCost;
        logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
        hpwl_state.log();

        node_die = data.node_die;
        node_pos = node_pos_sw;
        data.node_size = node_size_sw;
        via_data.node_size = node_size_sw.index({Slice(cell_mov_rhs, None)});
    }

    /* post process */  // TODO:
    if (st::setting.pp) {
        logger.info("setting.pp");
        torch::Tensor via_node_pos_init = node_pos.index({Slice(cell_mov_rhs, mov_rhs)});
        torch::Tensor via_node_pos_pp = via_node_pos_init.clone();
        torch::Tensor via_node_size_pp = via_data.node_size;
        torch::Tensor via_node_weight = via_data.bonding_map.clone();
        post_process(data,
                     node_pos,
                     data.node_die,
                     via_node_pos_pp,
                     via_node_size_pp,
                     via_data.die_info[2],
                     via_data.numRows,
                     via_data.row_height);
        auto node_pos_pp = torch::cat({node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_node_pos_pp}, 0);

        auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_pp.to(device), data.node_die.to(device), data);
        logger.info("");
        /*
        logger.info("After Post Process, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
        */
        logger.info("After Post Process, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        int via_cost = calc_via_cost(node_die, data) * data.bondingCost;
        logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
        hpwl_state.log();
        node_pos = node_pos_pp;

        printlog(
            LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    }

    /* via dp */  // TODO:
    if (st::setting.pp) {
        // node_pos = mov_node_pos_all.index({Slice(cell_mov_lhs, cell_mov_rhs)});  // FIXME: cell_mov_index
        torch::Tensor via_node_pos_init = node_pos.index({Slice(cell_mov_rhs, mov_rhs)});
        torch::Tensor via_node_pos = via_node_pos_init.clone();
        torch::Tensor via_node_pos_dp = via_node_pos.clone();
        torch::Tensor via_node_size_dp = via_data.node_size;
        torch::Tensor via_node_weight = via_data.bonding_map.clone();

        logger.info("================== NOW COMES THE DETAILED PLACEMENT ==================");
        viaDetailedPlace(via_node_pos.contiguous(),
                         via_node_size_dp,
                         via_node_pos_dp,
                         via_node_weight,
                         via_data.core_info,
                         via_data.numRows,
                         via_data.row_height,
                         via_node_pos.size(0),
                         node_pos,
                         data.node_die,
                         data);
        logger.info("================ DETAILED PLACEMENT DONE ==================");

        // conn_node_pos = torch::cat({node_pos_dp, via_node_pos_dp}, 0);
        auto node_pos_dp = torch::cat({node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}), via_node_pos_dp}, 0);

        auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_dp.to(device), data.node_die.to(device), data);
        logger.info("");
        /*
        logger.info("After VIA Post Process, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
        */
        logger.info("After VIA Post Process, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        hpwl_state.hpwls[hpwl_state.hpwl_idx++] = static_cast<long>(hpwl1.item<float>()) + static_cast<long>(hpwl2.item<float>());
        hpwl_state.hpwls_pp = hpwl_state.hpwls[hpwl_state.hpwl_idx - 1];
        hpwl_state.log();

        node_pos = node_pos_dp;

        printlog(
            LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    }

    /* dump to output FIXME: */
    int hpwls_final_ovhd = 0;
    hpwl_state.hpwls_raw = 0;
    // if (true) {
    if (st::setting.dp) {
        via_data.dump(node_pos, node_size, data.node_die, cell_mov_lhs, cell_mov_rhs, data.node_orient_top);

        // auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos, data.node_die, data);
        auto [hpwl1, hpwl2, hpwl_ovlp] =
            wa_wirelength_hpwl::get_hpwl_formatted(data, node_pos.to(device), data.node_die.to(device));
        // logWireLength(data, node_pos.to(device), data.node_die.to(device));
        auto [tmp1, tmp2, hpwl_ovhd, via_bbox] =
            evaluate_wl_ovhd_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        /*
        logger.notice("Final sol, exact HPWL [bot + top = total / overlap / overhead]: [%d + %d = %d / %d / %d]",
                      hpwl1.item<int>(), hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>(), hpwl_ovlp.item<int>(),
                      hpwl_ovhd.item<int>());
        */
        logger.notice(
            "Final sol, exact HPWL [bot + top = total / overlap / overhead]: [%.2f + %.2f = %.2f / %.2f / %.2f]",
            hpwl1.item<float>(),
            hpwl2.item<float>(),
            (hpwl1 + hpwl2).item<float>(),
            hpwl_ovlp.item<float>(),
            hpwl_ovhd.item<float>());
        /*
        logger.info("After formating, solution eval, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
        */
        logger.info("After formating, solution eval, exact HPWL [bot + top = total]: [%.2f + %.2f = %.2f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        // hpwl_state.hpwls_dp = (hpwl1 + hpwl2).item<int>();

        // int via_cost = calc_via_cost(node_die, data) * data.bondingCost;
        // logger.info("via cost = %d, total = %f", via_cost, (hpwl1 + hpwl2).item<float>()+via_cost);


        data.node_die.index({Slice(cell_mov_rhs, None)}) = -1;
        std::tie(hpwl1, hpwl2, hpwl_ovlp) = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        /*
        logger.info("Removing Vias, solution eval, exact HPWL [bot + top = total]: [%d + %d = %d]", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
        */
        logger.info("Removing Vias, solution eval, exact HPWL [bot + top = total]: [%f + %f = %f]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());
        // hpwl_state.hpwls_raw = (hpwl1 + hpwl2).item<int>();

        printlog(
            LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());
    }
    rawdb->writeOpenroad_vias(st::setting.output_path);

    std::string output_def = st::setting.output_path + "_bot.def";
    auto node_die_def = 1 - data.node_die.slice(0, data.cell_mov_lhs, data.cell_mov_rhs);
    std::vector<int> node_selected(node_die_def.data_ptr<int>(),
                                   node_die_def.data_ptr<int>() + node_die_def.size(0));
    rawdb->write_Openroad(st::setting.load_def_template, output_def, node_selected);
    output_def = st::setting.output_path + "_top.def";
    auto node_die_def2 = data.node_die.slice(0, data.cell_mov_lhs, data.cell_mov_rhs);
    std::vector<int> node_selected2(node_die_def2.data_ptr<int>(), node_die_def2.data_ptr<int>() + node_die_def2.size(0));
    rawdb->write_Openroad(st::setting.load_def_template, output_def, node_selected2);

#define EXTRACT(x, a, b, c)             \
    do {                                \
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

    if (true) {
        /* dump to csv */
        // std::filesystem::path out_log(std::string("./script_outputs/" + st::setting.design_name +
        // "_output_log.csv"));
        std::filesystem::path outlog_root(std::string("./script_outputs"));
        if (!std::filesystem::exists(outlog_root)) {
            std::filesystem::create_directories(outlog_root);
        }
        std::filesystem::path out_log(std::string(st::setting.design_name + "_output_log.csv"));
        std::filesystem::path output_log_file = outlog_root / out_log;

        ifstream infile;
        infile.open(output_log_file.string());
        // infile.open("./script_outputs/output_log.csv");
        bool is_empty = infile.peek() == std::ifstream::traits_type::eof();

        ofstream outfile;
        outfile.open(output_log_file.string(), std::ios_base::app);

        string tag = "";
        string value = "";
        string quote = ",";
        string newline = "\n";
        EXTRACT(st::setting.design_name, tag, value, quote);
        EXTRACT(via_data.num_bonds, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_gp, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_lg, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_dp, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_pp, tag, value, quote);
        EXTRACT(hpwls_final_ovhd, tag, value, quote);
        EXTRACT(hpwl_state.hpwls_raw, tag, value, quote);
        EXTRACT(st::setting.num_bin_x, tag, value, quote);
        EXTRACT(st::setting.num_bin_y, tag, value, quote);
        EXTRACT(st::setting.block_row, tag, value, quote);
        EXTRACT(st::setting.sideline, tag, value, quote);
        EXTRACT(st::setting.stack_cells, tag, value, quote);
        EXTRACT(st::setting.quad_penalty, tag, value, quote);
        EXTRACT(st::setting.quad_coeff, tag, value, quote);
        EXTRACT(st::setting.num_bin_3d, tag, value, quote);
        EXTRACT(st::setting.cut_net_thres, tag, value, quote);
        EXTRACT(st::setting.net_weight_coef, tag, value, quote);
        EXTRACT(st::setting.net_weight_offset, tag, value, quote);
        EXTRACT(st::setting.stop_overflow_3d, tag, value, quote);
        EXTRACT(st::setting.num_den_layer, tag, value, quote);
        EXTRACT(st::setting.round_recursion, tag, value, quote);
        EXTRACT(st::setting.net_cut_str_thrs, tag, value, quote);
        EXTRACT(st::setting.strengthen_via_density, tag, value, quote);
        EXTRACT(st::setting.strengthen_net_wa_coef, tag, value, quote);
        EXTRACT(st::setting.local_density_weight, tag, value, quote);
        EXTRACT(st::setting.wa_z_model, tag, value, quote);
        EXTRACT(st::setting.force_coeff_2d, tag, value, quote);
        EXTRACT(st::setting.wa_coeff_wa_z, tag, value, quote);
        EXTRACT(st::setting.wa_coeff_wa_xy, tag, value, quote);
        EXTRACT(st::setting.density_weight, tag, value, quote);
        EXTRACT(st::setting.wa_coeff, tag, value, quote);
        EXTRACT(st::setting.density_weight_coef, tag, value, quote);
        EXTRACT(st::setting.magic_hpwl, tag, value, newline);

        if (is_empty) {
            outfile << tag;
        }
        outfile << value;
    }
}
