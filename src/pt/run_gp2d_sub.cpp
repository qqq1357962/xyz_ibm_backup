
#include "partition.h"
#include "patoh.h"

void Partitioner::run_patoh_sub_grid(NodeData &data, torch::Tensor node_pos) {
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

    int grid_window = st::setting.omni_int;
    auto window_info = data.die_info / grid_window;

    /* Slice folds TODO: */
    double num_grids = st::setting.num_grids;
    int num_grids_int = ceil(num_grids);
    int num_const = num_grids_int * num_grids_int + st::setting.global_const;  // TODO:

    /* pt info */
    node_die = -torch::ones(num_nodes, torch::dtype(torch::kInt));
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));

    /* TODO: dynamic ratio update */
    double init_ratio = data.tech_ratio.item<double>();
    double ratio = init_ratio;
    for (int iw = 0; iw < grid_window; iw++) {
        for (int jw = 0; jw < grid_window; jw++) {
            auto x_offset = data.die_info[1] * (1 / (float)grid_window * iw);
            auto y_offset = data.die_info[3] * (1 / (float)grid_window * jw);

            auto x_l_w = x_offset;
            auto y_l_w = y_offset;
            auto x_h_w = data.die_info[1] * (1 / (float)grid_window * (iw + 1));
            auto y_h_w = data.die_info[3] * (1 / (float)grid_window * (jw + 1));

            auto parter_x_w_0 = (node_pos.index({"...", 0}) >= x_l_w);
            auto parter_x_w_1 = (node_pos.index({"...", 0}) < x_h_w);
            auto parter_y_w_0 = (node_pos.index({"...", 1}) >= y_l_w);
            auto parter_y_w_1 = (node_pos.index({"...", 1}) < y_h_w);

            /* select the grid and map the nodes */
            torch::Tensor sub_net_idx = torch::zeros(num_nodes, torch::dtype(torch::kInt));
            auto parter_w = parter_x_w_0 * parter_x_w_1 * parter_y_w_0 * parter_y_w_1;
            auto node_selector_w = torch::_cast_Int(parter_w);
            vector<int> map_back;
            for (int k = 0; k < num_nodes; k++) {
                if (node_selector_w[k].item<int>() == 1) {
                    map_back.push_back(k);
                    sub_net_idx[k] = 1;
                }
            }

            /* assign multi-weights */
            torch::Tensor node_grid = -torch::ones(num_nodes, torch::dtype(torch::kInt));
            for (int idx = 0; idx < num_grids_int; idx++) {
                for (int jdx = 0; jdx < num_grids_int; jdx++) {
                    int grid_idx = idx * num_grids_int + jdx;

                    auto x_l = x_offset + window_info[1] * (1 / num_grids * idx);
                    auto x_h = x_offset + window_info[1] * std::min(1 / num_grids * (idx + 1), 1.0);
                    auto y_l = y_offset + window_info[3] * (1 / num_grids * jdx);
                    auto y_h = y_offset + window_info[3] * std::min(1 / num_grids * (jdx + 1), 1.0);

                    auto parter_x_0 = (node_pos.index({"...", 0}) >= x_l);
                    auto parter_x_1 = (node_pos.index({"...", 0}) < x_h);
                    auto parter_y_0 = (node_pos.index({"...", 1}) >= y_l);
                    auto parter_y_1 = (node_pos.index({"...", 1}) < y_h);

                    /* select the grid and map the nodes */
                    auto parter = parter_x_0 * parter_x_1 * parter_y_0 * parter_y_1;

                    node_grid.index_put_({parter}, grid_idx);
                }
            }

            /* config partial PaToH node weight */
            int nNode = nodes.size();
            int *cwghts = new int[nNode * num_const];
            for (int i = 0; i < nNode; i++) {
                for (int j = 0; j < num_const; j++) {
                    if (j == node_grid[i].item<int>())
                        cwghts[i * num_const + j] = 1;
                    else
                        cwghts[i * num_const + j] = 0;
                }
                if (st::setting.global_const != 0) cwghts[i * num_const + num_const - 1] = 1;
            }

            /* config partial PaToH net weight */
            int nNets = nets.size();
            int *nwghts = new int[nNets];
            for (int i = 0; i < nNets; i++) {
                Box box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max());

                int64_t start_idx = 0;
                if (i != 0) {
                    start_idx = data.hyperedge_list_end[i - 1].item<int64_t>();
                }
                int64_t end_idx = data.hyperedge_list_end[i].item<int64_t>();
                if (end_idx != start_idx) {
                    for (int64_t idx = start_idx; idx < end_idx; idx++) {
                        int64_t pin_id = data.hyperedge_list[idx].item<int64_t>();
                        int64_t node_id = data.pin_id2node_id[pin_id].item<int64_t>();

                        box.xl = std::min(box.xl, (node_pos[node_id][0] + data.pin_rel_cpos[pin_id][0]).item<float>());
                        box.xh = std::max(box.xh, (node_pos[node_id][0] + data.pin_rel_cpos[pin_id][0]).item<float>());
                        box.yl = std::min(box.yl, (node_pos[node_id][1] + data.pin_rel_cpos[pin_id][1]).item<float>());
                        box.yh = std::max(box.yh, (node_pos[node_id][1] + data.pin_rel_cpos[pin_id][1]).item<float>());
                    }
                }

                float xc = box.center_x();
                float yc = box.center_y();

                auto x_l = data.die_info[1] * (1 / num_grids * 3);
                auto x_h = data.die_info[1] * std::min(1 / num_grids * (5 + 1), 1.0);
                auto y_l = data.die_info[3] * (1 / num_grids * 3);
                auto y_h = data.die_info[3] * std::min(1 / num_grids * (5 + 1), 1.0);

                auto parter_x_0 = (xc >= x_l);
                auto parter_x_1 = (xc < x_h);
                auto parter_y_0 = (xc >= y_l);
                auto parter_y_1 = (xc < y_h);
                bool parter = (parter_x_0 * parter_x_1 * parter_y_0 * parter_y_1).item<bool>();
                if (parter) nwghts[i] = st::setting.net_weight_offset;
                else {
                    if (net_to_num_pins[i].item<int>() <= st::setting.cut_net_thres)
                        nwghts[i] = (int)(st::setting.net_weight_coef * st::setting.net_weight_offset);
                    else
                        nwghts[i] = st::setting.net_weight_offset;
                }
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

            // /* construct graph */
            // /* count #pins */
            // for (int i = 0; i < _n; i++) {
            //     for (auto node : net->Nodes) {
            //         if (cwghts[node->id] == 1) {
            //             nPin++;
            //         }
            //     }
            // }

            // xpins = new int[_n + 1];
            // pins = new int[nPin];
            // for (int i = 0, p = 0; i < _n; i++) {
            //     auto net = nets[i];
            //     for (auto node : net->Nodes) {
            //         if (cwghts[node->id] == 1)
            //             pins[p++] = node->id;  
            //     }
            //     xpins[i] = p;
            // }
            // xpins[_n] = nPin;

            /* PaToH args */
            PaToH_Parameters args;
            // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
            PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
            args.seed = 0;
            args._k = numPart;
            args.final_imbal = st::setting.pt_imbl;
            int cut;

            args.MemMul_CellNet = 1000;
            args.MemMul_Pins = 1000;
            PaToH_Check_User_Parameters(&args, true);

            int *partvec = new int[nNode];
            for (int i = 0; i < nNode; i++) partvec[i] = -1;

            int *partweights = new int[numPart * _nconst];
            float *targetweights = new float[numPart * _nconst];

            for (int i = 0; i < _nconst; i++) {
                targetweights[_nconst * 0 + i] = ratio;
                targetweights[_nconst * 1 + i] = 1 - ratio;
            }

            PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);
            logger.info("Partitioner::run_patoh");
            PaToH_Part(&args, _c, _n, _nconst, 0, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights,
                       &cut);

            // TODO: dynamic ratio
            for (int i = 0; i < map_back.size(); i++) {
                int node_id = map_back[i];
                int group = partvec[node_id];
                node_die[node_id] = group;
                mov_cell_areas[group] += nodes[node_id]->sizes[group];
            }

            logger.info("Weights < %.2f <-- %.2f >", ratio,
                        ((1 - node_die) * node_selector_w).sum().item<float>() / node_selector_w.sum().item<float>());

            torch::Tensor mov_cell_util_top =
                ((max_mov_cell_areas[1] - mov_cell_areas[1]) / max_mov_cell_areas[1]).clamp(0.01, 1.0);
            torch::Tensor mov_cell_util_bot =
                ((max_mov_cell_areas[0] - mov_cell_areas[0]) / max_mov_cell_areas[0]).clamp(0.01, 1.0);
            float tech_ratio_sub = (mov_cell_util_bot / mov_cell_util_top).item<float>();
            logger.info("Partition ratio ajusted from %.2f -> %.2f", ratio, init_ratio * tech_ratio_sub);
            ratio = max(min(tech_ratio_sub * init_ratio, 1.0), 0.0);

            free(cwghts);
            free(nwghts);
            free(xpins);
            free(pins);
            free(partweights);
            free(partvec);
            PaToH_Free();
        }
    }
    
    if (true) {
        /* Slice folds TODO: */
        double num_grids = st::setting.num_grids;
        int num_grids_int = ceil(num_grids);
        int num_const = num_grids_int * num_grids_int;  // TODO:

        /* create local bins */
        node_grid = -torch::ones(num_nodes, torch::dtype(torch::kInt));
        max_grid_mov_cell_areas = torch::zeros({num_const, 2}, dtype(mov_cell_areas.dtype()));
        grid_mov_cell_areas = torch::zeros({num_const, 2}, dtype(mov_cell_areas.dtype()));
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
                parter = torch::_cast_Int(parter);
                float num_local_nodes = parter.sum().item<float>();

                /* max util */
                auto total_area_local_bot = (data.node_area_bot * parter).sum();
                auto total_area_local_top = (data.node_area_top * parter).sum();
                max_grid_mov_cell_areas[grid_idx][0] =
                    data.tech_ratio * total_area_local_bot * ((2 + num_local_nodes) / num_local_nodes);
                max_grid_mov_cell_areas[grid_idx][1] =
                    (1 - data.tech_ratio) * total_area_local_top * ((2 + num_local_nodes) / num_local_nodes);
                max_grid_mov_cell_areas[grid_idx][0] = data.area_ratio_bot * total_area_local_bot;
                max_grid_mov_cell_areas[grid_idx][1] = data.area_ratio_top * total_area_local_top;

                /* actual util */
                auto mov_cell_area_local_bot = (data.node_area_bot * parter * (1 - node_die)).sum();
                auto mov_cell_area_local_top = (data.node_area_top * parter * (node_die)).sum();
                grid_mov_cell_areas[grid_idx][0] = mov_cell_area_local_bot;
                grid_mov_cell_areas[grid_idx][1] = mov_cell_area_local_top;

                logger.info("Grid-%d Utils for each chip (%.2f, %.2f), tech_ratio %.2f", grid_idx,
                            (grid_mov_cell_areas[grid_idx][0] / max_grid_mov_cell_areas[grid_idx][0]).item<float>(),
                            (grid_mov_cell_areas[grid_idx][1] / max_grid_mov_cell_areas[grid_idx][1]).item<float>(),
                            (max_grid_mov_cell_areas[grid_idx][1] / max_grid_mov_cell_areas[grid_idx][0]).item<float>());
            }
        }

        /* validate local utilization constraints */
        grid_mov_cell_areas = torch::zeros({num_const, 2}, dtype(mov_cell_areas.dtype()));
        mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
        for (int i = 0; i < num_nodes; i++) {
            int group = node_die[i].item<int>();
            int grid_idx = node_grid[i].item<int>();

            if ((grid_mov_cell_areas[grid_idx][group] + nodes[i]->sizes[group] > max_grid_mov_cell_areas[grid_idx][group])
                    .item<bool>()) {
                group = !group;
            }

            grid_mov_cell_areas[grid_idx][group] += nodes[i]->sizes[group];
            mov_cell_areas[group] += nodes[i]->sizes[group];
            nodes[i]->group = group;
            node_die[i] = group;
        }
    }

    logger.info("============ Legalized partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)", (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());
}
