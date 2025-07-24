
#include "partition.h"
#include "patoh.h"
// #include "placer/run_placement.h"

torch::Tensor Partitioner::run_gp2d_grid(NodeData &data) {
    st::setting.cache_macro_mask = data.macro_mask.to(data.device);
    logger.info("============= Skip GP2D ============");

    // ======================================================================================================
    //
    //                                            GP2D
    //
    // ======================================================================================================
    /* initialization */
    st::setting.use_filler = st::setting.partitioner == "gp2d" ? false : st::setting.use_filler;
    st::setting.early_stop_check_plateau = true;

    logger.info("Optimizer info: density_weight: %3E | wa_coef %.2f | magic_hpwl %d", st::setting.density_weight,
                st::setting.wa_coeff, st::setting.magic_hpwl);

    torch::Tensor node_pos;
    data.init_density_map = torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype()));
    data.init_filler();
    auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info();
    // auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info_dummy_via();
    auto [mov_lhs, mov_rhs] = data.movable_index;

    // mov_node_pos[0][0] = 15;
    // mov_node_pos[0][1] = 15;

    // mov_node_pos = mov_node_pos.requires_grad_(true);
    mov_node_pos = mov_node_pos.to(data.device).requires_grad_(true);  
    mov_node_size = mov_node_size.to(data.device);                     
    expand_ratio = expand_ratio.to(data.device);                       
    data.to(data.device);
    // node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
    // mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    // for (int i = 0; i < num_nodes; i++) {
    //     int group = node_die[i].item<int>();
    //     if (st::setting.clamp_util) {
    //         if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
    //             group = !group;
    //     }
    //     mov_cell_areas[group] += nodes[i]->sizes[group];
    //     nodes[i]->group = group;
    //     node_die[i] = group;
    // }
    // node_die.to(device);
    // data.die_info.to(device);
    // data.node_die = node_die.clone();  // TODO: construct data from pt
    // data.mov_cell_areas = mov_cell_areas.clone();
    // return node_pos;

    torch::Tensor init_density_map = get_init_density_map(data);
    data.init_shape_params(mov_node_size);

    if (st::setting.skip_patoh) {
        data.to(torch::kCPU);
        return mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
    }

//     if (st::setting.pt_model == "null") { /* disable fillers to smoothen the distribution */
//         /* trunc nodes to core */
//         torch::Tensor node_pos_lb = mov_node_size / 2 + data.mov_node_sideline_ll + 1e-4;
//         torch::Tensor node_pos_ub = data.mov_node_sideline_ur - mov_node_size / 2 + data.die_ll - 1e-4;
//         // torch::Tensor node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4;
//         // torch::Tensor node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4;
//         std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn = [&node_pos_lb, &node_pos_ub](torch::Tensor x) {
//             x.data().clamp_(node_pos_lb, node_pos_ub);
//             return x;
//         };

//         /* overflow function */
//         std::function<torch::Tensor(torch::Tensor)> overflow_fn = [&data](torch::Tensor mov_density_map) {
//             torch::Tensor overflow_sum = ((mov_density_map - data.target_density) * data.bin_area).clamp_(0.0).sum();
//             return overflow_sum / data.__total_mov_area_without_filler__;
//         };
//         auto overflow_helper = make_tuple(mov_lhs, mov_rhs, overflow_fn);

//         /* parameteer scheduler */
//         ParamScheduler ps = ParamScheduler(data);
//         auto density_map_layer = ElectronicDensityLayer(data.unit_len, data.num_bin_x, data.num_bin_y, data.device,
//                                                         overflow_helper, expand_ratio, data.sorted_maps, data.macro_mask);

//         /* objective function */
//         /* follow the order of node types */
//         /* | MovConnected | MovFloat | ConnFixed ... | */
//         torch::Tensor conn_fix_node_pos = data.node_pos.new_empty({0, 2});
//         if (get<0>(data.fixed_connected_index) < get<1>(data.fixed_connected_index)) {
//             auto [lhs, rhs] = data.fixed_connected_index;
//             conn_fix_node_pos = data.node_pos.index({Slice(lhs, rhs), "..."});
//         }
//         conn_fix_node_pos = conn_fix_node_pos.detach();
//         std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> obj_and_grad_fn =
//             [&trunc_node_pos_fn, &mov_node_size, &init_density_map, &density_map_layer, &conn_fix_node_pos, &ps,
//              &data](at::Tensor mov_node_pos) {
//                 auto [loss, grad] = calc_obj_and_grad(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_map,
//                                          density_map_layer, conn_fix_node_pos, ps, data);
//                 if (true) {
//                     grad.index({torch::indexing::Slice(data.iopin_mov_lhs, data.iopin_mov_rhs), torch::indexing::Slice(0, 2)}) = 0.0;
//                 }
//                 return std::make_tuple(loss, grad);
//             };

//         /* evaluation function */
//         std::function<tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> evaluator_fn =
//             [&trunc_node_pos_fn, &mov_node_size, &init_density_map, &density_map_layer, &conn_fix_node_pos, &ps,
//              &data](at::Tensor mov_node_pos) {
//                 return fast_evaluator(mov_node_pos, trunc_node_pos_fn, mov_node_size, init_density_map,
//                                       density_map_layer, conn_fix_node_pos, ps, data);
//             };

//         // data.mov_node_weight = torch::ones({2}, dtype(torch::kFloat)).to(device);
//         // data.mov_node_weight[1] = -1;
//         // cout << mov_node_size << endl;

//         /* Nesterov optimizer */
//         auto optimizer = torch::optim::Nesterov({mov_node_pos}, torch::optim::NesterovOptions(0.0), obj_and_grad_fn);
//         init_params(mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, density_map_layer,
//                     mov_node_size, init_density_map, optimizer, ps, data);

//         auto pos_ret = mov_node_pos.clone();

//         /* learning rate */
//         double init_lr =
//             estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, st::setting.lr);
//         logger.info("Init learning rate %.3E", init_lr);
//         for (auto &group : optimizer.param_groups()) {
//             auto &options = static_cast<torch::optim::NesterovOptions &>(group.options());
//             options.set_lr(init_lr);
//         }

        

//         // data.mov_node_weight[1] = -1;
//         // cout << mov_node_size << endl;
//         // auto den_val_list = density_map_layer.forward(mov_node_pos, mov_node_size, init_density_map, data.mov_node_weight);
//         // exit(1);

//         /* start gp iteration */
//         printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(),
//                  utils::mem_use::get_peak());
//         logger.info("=========================================");
//         logger.info("start gp");
//         int &iteration = st::setting.iteration;
//         iteration = 0;  // FIXME: 0 ? 1
//         for (iteration = 0; iteration < 0; iteration++) {
//             torch::Tensor obj = optimizer.step();
//             // auto new_orient = data.node_orient_top.clone().contiguous();
//             if(st::setting.enable_rotate_in_gp)
//             {
//                 data.updata_shape_by_density_grad(st::setting.cache_density_grad_4part, mov_node_size, iteration);
//                 // data.update_macro_orientaion_by_pin_std(mov_node_pos);
//             }
//             auto [hpwl, overflow] = evaluator_fn(mov_node_pos);
//             ps.step(hpwl.item().toFloat(), overflow.item().toFloat(), mov_node_pos);
//             if (iteration % st::setting.log_freq == 0 || iteration == st::setting.inner_iter - 1 ||
//                 (iteration >= st::setting.minGPStep && ps.need_to_early_stop()) ) {
//                 logger.info(
//                     "iter: %d | masked_hpwl: %.2E overflow: %.4f obj: %.4E "
//                     "density_weight: %.4E wa_coeff: %.4E",
//                     iteration, hpwl.item().toFloat(), overflow.item().toFloat(), obj.item().toFloat(),
//                     ps.density_weight, ps.wa_coeff);
//                 // for(int ii = 0;ii<data.macro_shape_ratio.size();ii++)
//                 // {
//                 //     cout<<data.macro_shape_ratio[ii]<<endl;
//                 // }
//                 if (st::setting.draw_placement) {
//                     auto info = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_2D_GP");
//                     draw_fig_with_cairo_cpp(mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU),
//                                             mov_node_size.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU), 
//                                             data, info);
//                     info = make_tuple(0, 0, data.design_name + "_2D_GP2");
//                     int debuggg=0;
//                     // draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 0, 10);    
//                 }
//             }


//             if (iteration >= st::setting.minGPStep && ps.need_to_early_stop()) {
//                 break;
//             }
//         }

//         /* retrieve best score and evaluate without filler */
//         auto [best_sol, best_hpwl, best_overflow, best_iteration] = ps.get_best_solution();
//         if (best_iteration > -1) {
//             logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f", iteration, best_hpwl, best_overflow);
//         } else {
//             logger.info("GP Stop! cannot find best solution");
//         }
//         if (best_sol.numel() != 0) {
//             mov_node_pos.data().copy_(best_sol);
//         }

//         node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
//         if (false) {
//             mov_node_size.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}) *= 1;
//             mov_node_size.index({Slice(data.cell_mov_rhs)}) *= 0.1;
//             auto info = make_tuple(0, 0, data.design_name + "_2D_GP");
//             auto tensor1 = mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
//             auto tensor2 = mov_node_size.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
//             draw_fig_with_cairo_cpp(tensor1, tensor2, data, info);
//             info = make_tuple(0, 0, data.design_name + "_2D_GP2");
//             int debuggg=0;
//             draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 0, 10);
//             // info = make_tuple(0, 0, data.design_name + "_2D_GP3");
//             // draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 2792, 3);
//             // info = make_tuple(0, 0, data.design_name + "_2D_GP4");
//             // draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 2792, 3);
//             // info = make_tuple(0, 0, data.design_name + "_2D_GP5");
//             // draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 2792, 5);
//             // info = make_tuple(0, 0, data.design_name + "_2D_GP6");
//             // draw_fig_with_cairo_cpp2(tensor1, tensor2, data, info, 2792, 6);
//         }

//         auto [hpwl, overflow] = evaluate_placement(mov_node_pos, density_map_layer, init_density_map, data);
//         logger.info("After GP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f itertation: %d", hpwl.item().toFloat(),
//                     overflow.item().toFloat(), best_iteration);
//         // hpwl_state.hpwls[hpwl_state.hpwl_idx++] = hpwl.item<int>(); // FIXME:
        
//         // /* save to .pt model */
//         // if (st::setting.save_model) {
//         //     std::filesystem::path current_dir(std::filesystem::current_path());
//         //     std::filesystem::path result_dir(st::setting.result_dir);
//         //     std::filesystem::path exp_id(st::setting.exp_id);
//         //     std::filesystem::path res_root = current_dir / result_dir / exp_id;
//         //     std::string model_dir = res_root.string() + "/gp_pt.pt";
//         //     logger.info("Save gp2d to %s", model_dir.c_str());
//         //     torch::save(node_pos, model_dir);
//         // }

//         if (st::setting.eval_params) {

// #define EXTRACT(x, a, b, c)         \
//   do                                \
//   {                                 \
//     std::string xs((#x));           \
//     std::size_t pos = xs.find("."); \
//     xs = xs.substr(pos + 1);        \
//     a += (xs + c);                  \
//     std::stringstream ss;           \
//     ss << (x);                      \
//     std::string o;                  \
//     ss >> (o);                      \
//     b += (o + c);                   \
//   } while (false)

//             /* dump to csv */
//             ifstream infile;
//             infile.open("./script_outputs/gp_params_output_log.csv");
//             bool is_empty = infile.peek() == std::ifstream::traits_type::eof();

//             ofstream outfile;
//             outfile.open("./script_outputs/gp_params_output_log.csv", std::ios_base::app);

//             int hpwl_v = hpwl.item<int>();
//             float overflow_v = overflow.item<float>();

//             string tag = "";
//             string value = "";
//             string quote = ",";
//             string newline = "\n";
//             EXTRACT(st::setting.design_name, tag, value, quote);
//             EXTRACT(st::setting.quad_penalty, tag, value, quote);
//             EXTRACT(st::setting.quad_coeff, tag, value, quote);
//             EXTRACT(st::setting.wa_coeff, tag, value, quote);
//             EXTRACT(st::setting.density_weight, tag, value, quote);
//             EXTRACT(st::setting.density_weight_coef, tag, value, quote);
//             EXTRACT(st::setting.magic_hpwl, tag, value, quote);
//             EXTRACT(hpwl_v, tag, value, quote);
//             EXTRACT(overflow_v, tag, value, newline);

//             cout << tag << endl;
//             cout << value << endl;

//             if (is_empty) {
//                 outfile << tag;
//             }
//             outfile << value;
//             outfile.close();

//             exit(1);
//         }
//     } else {
//         logger.info("Loading model from %s", st::setting.pt_model.c_str());
//         torch::load(node_pos, st::setting.pt_model);
//     }

    data.to(torch::kCPU);
    // data.reset();  // FIXME:
    // run_patoh_sub_grid(data, node_pos.to(torch::kCPU));
    // run_patoh_grided(data, node_pos.to(torch::kCPU));
    // run_patoh(data, true);
    // run_patoh(data, false);
    run_patoh_area(data);

    return node_pos;
}


void Partitioner::run_patoh_grided(NodeData &data, torch::Tensor node_pos) {
    // ======================================================================================================
    //
    //                                            PARTITION
    //
    // ======================================================================================================
    logger.info("Propagating external pins: %d", st::setting.pin_propagation);
    logger.info("Dynamic ratio %d", st::setting.dynamic_ratio);
    logger.info("Mononlithic %d", st::setting.mononlithic);
    logger.info("#Grids %.1f", st::setting.num_grids);
    logger.info("#imbl %.3f", st::setting.pt_imbl);

    /* Slice folds TODO: */
    double num_grids = st::setting.num_grids;
    int num_grids_int = ceil(num_grids);

    /* TODO: dynamic ratio update */
    double init_ratio = data.tech_ratio.item<double>();
    double ratio = init_ratio;
    vector<vector<int>> sub_nets;
    sub_nets.resize(num_nets);
    node_die = -torch::ones(num_nodes, torch::dtype(torch::kInt));
    vector<int> num_cuts(num_grids_int * num_grids_int, 0);
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int idx = 0; idx < num_grids_int; idx++) {
        for (int jdx = 0; jdx < num_grids_int; jdx++) {
            auto x_l = data.die_info[1] * (1 / num_grids * idx);
            auto x_h = data.die_info[1] * std::min(1 / num_grids * (idx + 1), 1.0);
            auto y_l = data.die_info[3] * (1 / num_grids * jdx);
            auto y_h = data.die_info[3] * std::min(1 / num_grids * (jdx + 1), 1.0);

            auto parter_x_0 = (node_pos.index({"...", 0}) >= x_l);
            auto parter_x_1 = (node_pos.index({"...", 0}) < x_h);
            auto parter_y_0 = (node_pos.index({"...", 1}) >= y_l);
            auto parter_y_1 = (node_pos.index({"...", 1}) < y_h);

            /* select the grid and map the nodes */
            auto parter = parter_x_0 * parter_x_1 * parter_y_0 * parter_y_1;
            auto node_selector = torch::_cast_Int(parter);
            vector<int> map_back;
            torch::Tensor sub_net_idx = torch::zeros(num_nodes, torch::dtype(torch::kInt));
            for (int k = 0; k < num_nodes; k++) {
                if (node_selector[k].item<int>() == 1) {
                    map_back.push_back(k);
                    sub_net_idx[k] = 1;
                }
            }
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            index_type num_idx = map_back.size();
            torch::Tensor mapper = torch::from_blob(map_back.data(), {num_idx}, options);

            /* run partial PaToH */
            int nNets = nets.size();
            int *nwghts = new int[nNets];
            for (int i = 0; i < nNets; i++) {
                if (net_to_num_pins[i].item<int>() <= st::setting.cut_net_thres)
                    nwghts[i] = (int)(st::setting.net_weight_coef * st::setting.net_weight_offset);
                else
                    nwghts[i] = st::setting.net_weight_offset;
                // nwghts[i] = (int)(check_net_weight(data, i, node_pos) * st::setting.net_weight_offset);
            }
            logger.info("Nets smaller than %d will be applied with wa force %.3f", st::setting.cut_net_thres,
                        st::setting.net_weight_coef);

            int nNode = nodes.size();
            int *cwghts = new int[nNode];
            for (int i = 0; i < nNode; i++) {
                cwghts[i] = (node_selector[i].item<int>() == 1) ? 1 : 0;
            }

            int _c = nNode;
            int _n = nNets;
            int _nconst = 1;
            int useFixCells = false;

            int nPin = 0;
            int *xpins;
            int *pins;

            if (st::setting.pin_propagation) {
                useFixCells = 1;
                /* count #pins */
                vector<bool> visiblility(num_nets, 0);
                for (int i = 0; i < _n; i++) {
                    auto net = nets[i];
                    bool visible = false;
                    for (auto node : net->Nodes) {
                        if (cwghts[node->id] == 1) {
                            nPin++;
                            visible = true;
                        }
                    }
                    visiblility[i] = visible;
                    if (visible) {
                        nPin += sub_nets[i].size();
                    }
                }

                /* add pins */
                xpins = new int[_n + 1];
                pins = new int[nPin];
                for (int i = 0, p = 0; i < _n; i++) {
                    if (visiblility[i]) {
                        for (auto node_id_sub_net : sub_nets[i]) {
                            pins[p++] = node_id_sub_net;
                            sub_net_idx[node_id_sub_net] = 1;
                        }
                    }
                    auto net = nets[i];
                    for (auto node : net->Nodes) {
                        if (cwghts[node->id] == 1) {
                            cout << "node id " << node->id << endl;
                            pins[p++] = node->id;
                            sub_nets[i].push_back(node->id);
                        }
                    }
                    xpins[i] = p;
                }
                xpins[_n] = nPin;

                if (false) {
                    torch::Tensor node_size_bot = torch::zeros({node_pos.size(0), 2}, torch::dtype(node_pos.dtype()));
                    torch::Tensor node_size_top = torch::zeros({node_pos.size(0), 2}, torch::dtype(node_pos.dtype()));
                    for (int i = 0; i != node_pos.size(0); i++) {
                        if (sub_net_idx[i].item<int>() == 0) {
                            node_size_bot[i][0] = data.node_size_bot[i][0];
                            node_size_bot[i][1] = data.node_size_bot[i][1];
                        } else {
                            node_size_top[i][0] = data.node_size_top[i][0];
                            node_size_top[i][1] = data.node_size_top[i][1];
                        }
                    }
                    auto info = make_tuple(idx * num_grids_int + jdx, 0, data.design_name + "_SUB_NET");
                    auto node_pos_draw_cp = torch::cat({node_pos, node_pos}, 0);
                    auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
                    draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info);
                }
            } else {
                useFixCells = false;

                for (int i = 0; i < _n; i++) {
                    auto net = nets[i];
                    for (auto node : net->Nodes) {
                        if (cwghts[node->id] == 1) nPin++;
                    }
                }

                xpins = new int[_n + 1];
                pins = new int[nPin];
                for (int i = 0, p = 0; i < _n; i++) {
                    auto net = nets[i];
                    for (auto node : net->Nodes) {
                        cout << "node id " << node->id << endl;
                        if (cwghts[node->id] == 1) pins[p++] = node->id;
                    }
                    xpins[i] = p;
                }
                xpins[_n] = nPin;
            }  // END IF

            /* PaToH args */
            PaToH_Parameters args;
            // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
            PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
            args.seed = 0;
            args._k = numPart;
            // args.final_imbal = 0.01;  // 0: completely ignore balance
            args.final_imbal = st::setting.pt_imbl;

            args.MemMul_Pins = 1000;
            PaToH_Check_User_Parameters(&args, true);

            int *partvec = new int[nNode];
            int *partweights = new int[numPart];
            int cut;
            PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

            for (int i = 0; i < nNode; i++) partvec[i] = node_die[i].item<int>();

            // logger.info("Partitioner::run_patoh");

            float *targetweights = new float[2];

            targetweights[0] = ratio;
            targetweights[1] = 1 - ratio;  // TODO:

            PaToH_Part(&args, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec,
                       partweights, &cut);

            logger.info("#pins: %d | %d cuts in sub-net-%d", nPin, cut, idx * num_grids_int + jdx);
            num_cuts.push_back(cut);

            for (int i = 0; i < nNode; i++) {
                if (node_selector[i].item<int>() == 1) {
                    int group = partvec[i];
                    node_die[i] = group;
                    nodes[i]->group = group;
                }
            }

            free(cwghts);
            free(nwghts);
            free(xpins);
            free(pins);
            free(partweights);
            free(partvec);
            PaToH_Free();

            // TODO: dynamic ratio
            for (int i = 0; i < map_back.size(); i++) {
                int node_id = map_back[i];
                int group = node_die[node_id].item<int>();
                mov_cell_areas[group] += nodes[node_id]->sizes[group];
            }

            if (st::setting.dynamic_ratio) {
                torch::Tensor mov_cell_util_top =
                    ((max_mov_cell_areas[1] - mov_cell_areas[1]) / max_mov_cell_areas[1]).clamp(0.01, 1.0);
                torch::Tensor mov_cell_util_bot =
                    ((max_mov_cell_areas[0] - mov_cell_areas[0]) / max_mov_cell_areas[0]).clamp(0.01, 1.0);
                float tech_ratio_sub = (mov_cell_util_bot / mov_cell_util_top).item<float>();
                logger.info("Partition ratio ajusted from %.2f -> %.2f", ratio, init_ratio * tech_ratio_sub);
                ratio = max(min(tech_ratio_sub * init_ratio, 1.0), 0.0);
            }
        }
    }
    int sum_of_cuts = std::accumulate(num_cuts.begin(), num_cuts.end(), 0);
    logger.info("%d cuts from inter-grid sub-nets", sum_of_cuts);

    logger.info("============ Original partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)", (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());
}

void Partitioner::run_patoh_mononlithic(NodeData &data, torch::Tensor node_pos) {
    // ======================================================================================================
    //
    //                                            PARTITION
    //
    // ======================================================================================================
    logger.info("Propagating external pins: %d", st::setting.pin_propagation);
    logger.info("Dynamic ratio %d", st::setting.dynamic_ratio);
    logger.info("Mononlithic %d", st::setting.mononlithic);
    logger.info("#Grids %.1f", st::setting.num_grids);
    logger.info("#imbl %.3f", st::setting.pt_imbl);

    /* Slice folds TODO: */
    double num_grids = st::setting.num_grids;
    int num_grids_int = ceil(num_grids);
    int num_const = num_grids_int * num_grids_int + st::setting.global_const;  // TODO:

    node_die = -torch::ones(num_nodes, torch::dtype(torch::kInt));
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));

    /* assign multi-weights */
    torch::Tensor node_grid = -torch::ones(num_nodes, torch::dtype(torch::kInt));
    int nNode = nodes.size();
    int *cwghts = new int[nNode * num_const];
    for (int idx = 0; idx < num_grids_int; idx++) {
        for (int jdx = 0; jdx < num_grids_int; jdx++) {
            int grid_idx = idx * num_grids_int + jdx;

            auto x_l = data.die_info[1] * (1 / num_grids * idx);
            auto x_h = data.die_info[1] * std::min(1 / num_grids * (idx + 1), 1.0);
            auto y_l = data.die_info[3] * (1 / num_grids * jdx);
            auto y_h = data.die_info[3] * std::min(1 / num_grids * (jdx + 1), 1.0);

            auto parter_x_0 = (node_pos.index({"...", 0}) >= x_l);
            auto parter_x_1 = (node_pos.index({"...", 0}) < x_h);
            auto parter_y_0 = (node_pos.index({"...", 1}) >= y_l);
            auto parter_y_1 = (node_pos.index({"...", 1}) < y_h);

            /* select the grid and map the nodes */
            auto parter = parter_x_0 * parter_x_1 * parter_y_0 * parter_y_1;

            node_grid.index_put_({parter}, grid_idx);
        }
    }

    /* run partial PaToH */
    for (int i = 0; i < nNode; i++) {
        for (int j = 0; j < num_const; j++) {
            if (j == node_grid[i].item<int>())
                cwghts[i * num_const + j] = 1;
            else
                cwghts[i * num_const + j] = 0;
        }
        if (st::setting.global_const != 0) cwghts[i * num_const + num_const - 1] = 1;
    }

    /* run partial PaToH */
    int nNets = nets.size();
    int *nwghts = new int[nNets];
    for (int i = 0; i < nNets; i++) {
        if (net_to_num_pins[i].item<int>() <= st::setting.cut_net_thres)
            nwghts[i] = (int)(st::setting.net_weight_coef * st::setting.net_weight_offset);
        else
            nwghts[i] = st::setting.net_weight_offset;
        // nwghts[i] = (int)(check_net_weight(data, i, node_pos) * st::setting.net_weight_offset);
    }

    /* PaToH configs */
    int _c = nNode;
    int _n = nNets;
    int _nconst = num_const;
    int useFixCells = 0;

    int nPin = 0;
    int *xpins;
    int *pins;

    /* construct graph */
    for (auto net : nets) {
        nPin += net->Nodes.size();
    }

    xpins = new int[_n + 1];
    pins = new int[nPin];
    for (int i = 0, p = 0; i < _n; i++) {
        auto net = nets[i];
        for (auto node : net->Nodes) {
            pins[p++] = node->id;
        }
        xpins[i] = p;
    }
    xpins[_n] = nPin;

    /* PaToH args */
    PaToH_Parameters args;
    // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
    PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
    args.seed = 0;
    args._k = numPart;
    // args.final_imbal = 0.01;  // 0: completely ignore balance
    // FIXME: imbl must be enabled
    args.final_imbal = st::setting.pt_imbl;
    int cut;

    args.MemMul_CellNet = 1000;
    args.MemMul_Pins = 1000;
    PaToH_Check_User_Parameters(&args, true);

    int *partvec = new int[nNode];
    for (int i = 0; i < nNode; i++) partvec[i] = -1;

    int *partweights = new int[numPart * _nconst];
    float *targetweights = new float[numPart * _nconst];

    double ratio = data.tech_ratio.item<double>() + st::setting.top_util_filler;
    for (int i = 0; i < _nconst; i++) {
        targetweights[_nconst * 0 + i] = ratio;
        targetweights[_nconst * 1 + i] = 1 - ratio;
    }

    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

    logger.info("Partitioner::run_patoh");
    PaToH_Part(&args, _c, _n, _nconst, 0, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, &cut);

    // for (int i = 0; i < _nconst; i++) {
    //     int a = partweights[numPart * i + 0];
    //     int b = partweights[numPart * i + 1];
    //     logger.info("#Grids -> (%d, %d)", partweights[numPart * i + 0], partweights[numPart * i + 1]);
    //     logger.info("Grid-%d weights -> (%.2f, %.2f)", i, (float)(a) / (a + b), (float)(b) / (a + b));
    // }
    for (int i = 0; i < _nconst; i++) {
        int a = partweights[_nconst * 0 + i];
        int b = partweights[_nconst * 1 + i];
        logger.info("Grid-%d weights -> (%.2f, %.2f)", i, (float)(a) / (a + b), (float)(b) / (a + b));
    }

    for (int i = 0; i < nNode; i++) {
        int group = partvec[i];
        node_die[i] = group;
        nodes[i]->group = group;
        mov_cell_areas[group] += nodes[i]->sizes[group];
    }

    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);
    free(partvec);
    PaToH_Free();

    logger.info("============ Original partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)", (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());
}