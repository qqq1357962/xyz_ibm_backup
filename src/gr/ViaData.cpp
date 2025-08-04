#include "ViaData.h"

ViaData::ViaData(NodeData& data_, shared_ptr<db::Database> rawdb_, torch::Tensor cell_die_)
    : data_ptr(make_shared<NodeData>(data_)), rawdb(rawdb_), device(data_.device) {
    auto data = *data_ptr;
    /* general info */
    // num_nodes = cell_pos_.size(0);
    cell_mov_lhs = data.cell_mov_lhs;
    cell_mov_rhs = data.cell_mov_rhs;
    iopin_mov_lhs = data.iopin_mov_lhs;
    iopin_mov_rhs = data.iopin_mov_rhs;
    bondingInfo = data.bondingInfo;
    num_nodes = data.num_nodes;
    num_nets = data.hyperedge_list_end.size(0);
    num_pins = data.hyperedge_list.size(0);
    cell_die_prime = cell_die_;
    // cell_pos_prime = cell_pos_;
    bin_area = data.bin_area;
    unit_len = data.unit_len;
    __die_scale__ = data.__die_scale__;
    site_width = data.site_width;
    site_width_current = data.site_width_current;
    site_height = data.site_height;
    num_bin_x = data.num_bin_x;
    num_bin_y = data.num_bin_y;

    /* FIXME: neet to consider to spacing rule between die boundaries */
    row_height = data.bondingInfo[1] + data.bondingInfo[2];
    /* TODO: not necessary to align bonding terminals to rows, but for simplification */
    numRows = torch::floor((data.die_info[3] - data.die_info[2] - data.bondingInfo[2]) / row_height);

    core_info = data.die_info.clone();
    die_info = data.die_info.clone();
    die_info_back_up = data.die_info_back_up.clone();
    core_info[0] += data.bondingInfo[2] / 2;  // TODO: safe bound for spacing rule
    core_info[1] -= data.bondingInfo[2] / 2;
    core_info[2] += data.bondingInfo[2] / 2;  // FIXME: used as a shift
    core_info[3] -= data.bondingInfo[2] / 2;
    // core_info[3] -= data.bondingInfo[2] / 2;
    die_ur = die_info.reshape({2, 2}).t()[1].clone();
    die_ll = die_info.reshape({2, 2}).t()[0].clone();
    core_ur = core_info.reshape({2, 2}).t()[1].clone();
    core_ll = core_info.reshape({2, 2}).t()[0].clone();

    /* Max utilizations */
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    max_mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    max_mov_cell_areas[0] = data.mov_cell_area_bot_bound;
    max_mov_cell_areas[1] = data.mov_cell_area_top_bound;

    /* reconstruct nets */
    node_to_num_pins = torch::ones({num_nets, 1}, torch::dtype(torch::kFloat));
    torch::Tensor net_cut_info = torch::zeros({num_nets, 2}, torch::dtype(torch::kInt));
    const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = data.pin_id2node_id.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = data.hyperedge_list.accessor<int64_t, 1>();
    const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = data.hyperedge_list_end.accessor<int64_t, 1>();
    const torch::TensorAccessor<int, 1> cell_die_at = cell_die_prime.accessor<int, 1>();
    for (int i = 0; i != num_nets; i++) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end_at[i - 1];
        }
        int64_t end_idx = hyperedge_list_end_at[i];
        if (end_idx != start_idx) {
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list_at[idx];
                int64_t node_id = pin_id2node_id_at[pin_id];
                int c_id = cell_die_at[node_id];

                net_cut_info[i][c_id] += 1;
            }
        }
    }

    
    /* node size (w+s, h+s) */
    /* node_die = -1 : invisible */
    /* node_die =  2 : calculated in both chips */
    /* net_mask = -1 : ignored in wl grad */
    bonding_map = torch::_cast_Int(torch::prod(net_cut_info, 1) != 0);
    num_bonds = bonding_map.sum().item<int>();
    logger.info("#bondings: %d", num_bonds);

#ifdef DEBUG
#ifdef WITH_OPENCV
    if (true) {
        torch::Tensor w = torch::ones(data.net_to_num_pins.sizes()[0]);
        pin_num_count_cut = torch::_cast_Int(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        pin_num_count_cut.scatter_add_(0, torch::_cast_Long(data.net_to_num_pins), torch::_cast_Int(bonding_map));

        if (true) {
            plt::backend("Agg");
            int view = 20;
            int n = min(view, data.net_to_num_pins.max().item<int>() + 1);
            std::vector<double> x(n), w(n), y(n);
            for (int i = 0; i < n - 1; ++i) {
                x.at(i) = i;
                y.at(i) = data.pin_num_count[i].item<int>();
                w.at(i) = pin_num_count_cut[i].item<int>();
            }
            x.at(n - 1) = n - 1;
            y.at(n - 1) = data.pin_num_count.index({Slice(view, None)}).sum().item<int>();
            w.at(n - 1) = pin_num_count_cut.index({Slice(view, None)}).sum().item<int>();

            plt::plot(x, y);
            plt::plot(x, w, "r");
            vector<int> xtick = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
            plt::xticks(xtick);

            std::filesystem::path current_dir(std::filesystem::current_path());
            std::filesystem::path result_dir(st::setting.result_dir);
            std::filesystem::path exp_id(st::setting.exp_id);
            std::filesystem::path res_root = current_dir / result_dir / exp_id;
            std::string fig_path = res_root.string() + "/" + "net_pin_num.png";
            plt::save(fig_path);
            plt::close();
        }
        plt::close();
    }
    if (true) {
        torch::Tensor w = torch::ones(data.node_to_num_pins.sizes()[0]);
        auto node_pin_num_count = torch::zeros(data.node_to_num_pins.max().item<int>() + 1);
        node_pin_num_count.scatter_add_(0, torch::_cast_Long(data.node_to_num_pins.index({"...", 0})), w);

        if (true) {
            plt::backend("Agg");
            int view = 20;
            int n = min(view, data.node_to_num_pins.max().item<int>() + 1);
            std::vector<double> x(n), w(n), y(n);
            for (int i = 0; i < n - 1; ++i) {
                x.at(i) = i;
                y.at(i) = node_pin_num_count[i].item<int>();

            }
            x.at(n - 1) = n - 1;
            y.at(n - 1) = node_pin_num_count.index({Slice(view, None)}).sum().item<int>();

            plt::plot(x, y);
            vector<int> xtick = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
            plt::xticks(xtick);

            std::filesystem::path current_dir(std::filesystem::current_path());
            std::filesystem::path result_dir(st::setting.result_dir);
            std::filesystem::path exp_id(st::setting.exp_id);
            std::filesystem::path res_root = current_dir / result_dir / exp_id;
            std::string fig_path = res_root.string() + "/" + "node_pin_num.png";
            plt::save(fig_path);
            plt::close();
        }
        plt::close();
    }
#endif // WITH_OPENCV
#endif //  DEBUG

    node_size = torch::zeros({num_nets, 2}, torch::dtype(torch::kFloat));
    node_die = torch::zeros(num_nets, torch::dtype(torch::kInt));
    net_mask = torch::ones(num_nets, torch::dtype(torch::kBool));
    for (int i = 0; i != num_nets; i++) {
        if (bonding_map[i].item<int>() == 0) {
            node_size[i][0] = 0;
            node_size[i][1] = 0;
            node_die[i] = -1;
            net_mask[i] = 0;
        } else {
            node_size[i][0] = data.bondingInfo[0] + data.bondingInfo[2];
            node_size[i][1] = data.bondingInfo[1] + data.bondingInfo[2];
            node_die[i] = 2;
        }
    }

    /* precompute_var */
    target_density = st::setting.via_target_density;
    use_filler = st::setting.num_den_layer == 3 ? true : false;
    node_type_indices.emplace_back(std::make_tuple(0, num_nets, "Mov"));

}  // END MODULE

//---------------------------------------------------------------------

void ViaData::postscale() {
    logger.info("design scaled by %d", site_width);
    site_width_current = site_width;
    die_info *= site_width;
    core_info *= site_width;
    row_height *= site_width;
    node_size *= site_width;
    // site_width = 1;
}  // END MODULE

//---------------------------------------------------------------------

void ViaData::init_vars() {
    auto data = *data_ptr;
    /* PlaceData::compute_filler */
    /* vias are all movalbe */
    movable_index = make_tuple(0, num_nets);
    auto [mov_lhs, mov_rhs] = movable_index;
    torch::Tensor mov_node_size = node_size;
    torch::Tensor die_area = torch::prod(core_ur - core_ll);

    at::Tensor single_sideline_ll = core_ll - 1e-4;
    at::Tensor single_sideline_ur = core_ur + core_ll - 1e-4;
    torch::Tensor single_sideline_space_ll = torch::tensor({(float)0.0, (float)0.0}, dtype(torch::kFloat));
    torch::Tensor single_sideline_space_ur = torch::tensor({(float)0.0, 
                                    (core_ur[1] - numRows * row_height - 1e-4).item<float>()}, dtype(torch::kFloat));
    // torch::Tensor single_sideline_space_ll = torch::tensor({bondingInfo[2].item<float>(), bondingInfo[2].item<float>()}, dtype(torch::kFloat));
    // torch::Tensor single_sideline_space_ur = torch::tensor({bondingInfo[2].item<float>(),  
    //                                 bondingInfo[2].item<float>() + (die_ur[1] - numRows * row_height - 1e-4).item<float>()}, dtype(torch::kFloat));

    mov_node_sideline_ll = single_sideline_ll.repeat({num_nets, 1});
    mov_node_sideline_ur = single_sideline_ur.repeat({num_nets, 1});

    mov_node_sideline_ll += single_sideline_space_ll; // FIXME:
    mov_node_sideline_ur -= single_sideline_space_ur;
        
    // torch::Tensor placeable_area = die_area;  // TODO: target_density
    torch::Tensor placeable_die_area = at::prod(die_ur - die_ll - single_sideline_space_ur);
    logger.info(
            "Placeable area of die/sidelined: [%.4E, %4E]", die_area.item<float>(), 
                                                            placeable_die_area.item<float>());
    mov_cell_area = torch::prod(mov_node_size, 1);
    __total_mov_area_without_filler__ = torch::sum(mov_cell_area).item<float>();

    if (use_filler) {
        int num_movable_nodes = mov_rhs - mov_lhs;
        at::Tensor mov_node_xsize_order = torch::argsort(mov_node_size.index({"...", 0}));

        // FIXME: filler size normalization
        at::Tensor filler_size_x = data.bondingInfo[0] + data.bondingInfo[2];
        at::Tensor filler_size_y = data.bondingInfo[1] + data.bondingInfo[2];

        at::Tensor total_filler_area = max(target_density * placeable_die_area - torch::sum(mov_cell_area),
                                           at::tensor(0.0, dtype(torch::kFloat).device(torch::kCPU)));
        single_filler_size = at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>()},
                                        torch::dtype(mov_node_size.dtype()).device(torch::kCPU));
        __num_fillers__ = torch::round(total_filler_area / (filler_size_x * filler_size_y)).item<int>();

        if (__num_fillers__ > 1e6) {
            logger.warning("Too many fillers: %d, set to 1000000", __num_fillers__);
            int scaler = static_cast<int>(ceil(sqrt(static_cast<double>(__num_fillers__) / 1000000.0)));
            filler_size_x *= scaler;
            filler_size_y *= scaler;
            single_filler_size = at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>()},
                                            torch::dtype(mov_node_size.dtype()).device(torch::kCPU));
            __num_fillers__ = torch::round(total_filler_area / (filler_size_x * filler_size_y)).item<int>();
        }

        mov_node_sideline_ll = torch::cat({mov_node_sideline_ll, 
                                        (single_sideline_ll).repeat({__num_fillers__, 1})}, 0);
        mov_node_sideline_ur = torch::cat({mov_node_sideline_ur, 
                                        (single_sideline_ur).repeat({__num_fillers__, 1})}, 0);
    }
    if (__num_fillers__ > 0) {
        filler_size = single_filler_size.repeat({__num_fillers__, 1});
        logger.info("Via GP #Fillers: %d Filler size: (%.4e, %.4e)",
                    __num_fillers__,
                    single_filler_size[0].item<float>(),
                    single_filler_size[1].item<float>());
    } else {
        logger.info(
            "Via GP stage: num_fillers[%d] is smaller or equal to 0. Please make sure target_density[%.2f]"
            " is larger than movable cell utilization[%.2f]. use_filler is disable.",
            __num_fillers__,
            target_density,
            torch::sum(torch::prod(mov_node_size, 1)).item<float>());
        use_filler = false;
    }

    /* PlaceData::pre_compute_var */
    node_area = torch::prod(node_size, 1).unsqueeze(1);
    mov_node_area = node_area;
    mov_node_to_num_pins = node_to_num_pins;
    if (use_filler) {
        at::Tensor filler_area = torch::prod(filler_size, 1).unsqueeze(1);
        mov_node_area = torch::cat({mov_node_area, filler_area}, 0);
        at::Tensor filler_to_num_pins = mov_node_to_num_pins.new_zeros({__num_fillers__, 1});
        assert(filler_to_num_pins.sizes() == filler_area.sizes());
        mov_node_to_num_pins = torch::cat({mov_node_to_num_pins, filler_to_num_pins}, 0);
    }

    /* PlaceData::compute_sorted_node_map */
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
        auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        sorted_maps =
            make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));
    }

    // /* update hyperedge lists */
    // /* via_mov_node | cell_node */
    // // pin_id2node_id -> cells_pin | vias_pin
    // // hyperedge_list -> net_pin | via
    // // hyperedge_list_end -> hyperedge_list_end + 1
    // pin_id2node_id = torch::zeros(num_pins + num_nets, torch::dtype(data.pin_id2node_id.dtype()));
    // hyperedge_list = torch::zeros(num_pins + num_nets, torch::dtype(data.hyperedge_list.dtype()));
    // hyperedge_list_end = torch::zeros(num_nets, torch::dtype(data.hyperedge_list.dtype()));
    // // hyperedge_list_end = torch::zeros_like(data.hyperedge_list_end);

    // // torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = pin_id2node_id.accessor<int64_t, 1>();
    // // torch::TensorAccessor<int64_t, 1> hyperedge_list_at = hyperedge_list.accessor<int64_t, 1>();
    // // torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = hyperedge_list_end.accessor<int64_t, 1>();
    // torch::TensorAccessor<int64_t, 1> __ori_pin_id2node_id_at = data.pin_id2node_id.accessor<int64_t, 1>();
    // torch::TensorAccessor<int64_t, 1> __ori_hyperedge_list_at = data.hyperedge_list.accessor<int64_t, 1>();
    // torch::TensorAccessor<int64_t, 1> __ori_hyperedge_list_end_at = data.hyperedge_list_end.accessor<int64_t, 1>();

    // /* update hyperedge lists */
    // index_type ptr = 0;
    // index_type last_idx = 0;
    // for (int i = 0; i < num_nets; ++i) {
    //     index_type start_idx = 0;
    //     if (i != 0) {
    //         start_idx = __ori_hyperedge_list_end_at[i - 1];
    //     }
    //     index_type end_idx = __ori_hyperedge_list_end_at[i];
    //     /* add cell pin */
    //     for (index_type idx = start_idx; idx < end_idx; idx++) {
    //         hyperedge_list[ptr] = __ori_hyperedge_list_at[idx];
    //         ptr++;
    //     }
    //     /* add via */
    //     hyperedge_list[ptr] = num_pins + i;
    //     ptr++;
    //     last_idx += end_idx - start_idx + 1;
    //     hyperedge_list_end[i] = last_idx;
    // }
    // /* update pin_id */
    // /* #pins -> #nets + #nodes */
    // for (int i = 0; i < num_pins + num_nets; ++i) {
    //     if (i < num_pins) {
    //         // pin_id2node_id[i] = data.pin_id2node_id[i];
    //         /* net pins */
    //         pin_id2node_id[i] = data.pin_id2node_id[i] + num_nets;  // FIXME: list length
    //     } else {
    //         // pin_id2node_id[i] = (i - num_pins + num_nodes);
    //         /* via pins */
    //         pin_id2node_id[i] = (i - num_pins);
    //     }
    // }

    /* update wirelength info */
    hpwl_scale = data.hpwl_scale;
    pin_rel_cpos = data.pin_rel_cpos;
    // const torch::TensorAccessor<int, 1> cell_die_at = cell_die_prime.accessor<int, 1>();
    // for (int i = 0; i < pin_rel_cpos.size(0); ++i) {  // FIXME: pin_rel
    //     long node_id = data.pin_id2node_id[i].item<long>();
    //     if (cell_die_at[node_id] == 0) {
    //         pin_rel_cpos[i] = data.pin_rel_cpos_bot[i];
    //     } else {
    //         pin_rel_cpos[i] = data.pin_rel_cpos_top[i];
    //     }
    // }
    auto via_pin_rel_cpos = torch::zeros({num_nets, 2}, torch::dtype(data.node_pos.dtype()));
    pin_rel_cpos = torch::cat({pin_rel_cpos, via_pin_rel_cpos}, 0);
}  // END MODULE

//---------------------------------------------------------------------

/// @brief follow PlaceData defination
/// @brief dump information to via hyperlist
/// @return return via move_node_info
tuple<torch::Tensor, torch::Tensor, torch::Tensor> ViaData::get_mov_node_info() {
    auto data = *data_ptr;
    /* via node info */
    node_pos = data.bonding_pos;
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor mov_node_pos = node_pos.index({Slice(mov_lhs, mov_rhs)}).clone();
    at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)}).clone();

    torch::Tensor scale = (data.die_ur - data.die_ll) * 0.001;
    torch::Tensor loc = (data.die_ur + data.die_ll) * 0.5;
    mov_node_pos = torch::randn_like(mov_node_pos) * scale + loc;
    if (use_filler) {
        at::Tensor filler_pos = torch::rand({__num_fillers__, 2}, torch::dtype(mov_node_size.dtype()));
        at::Tensor scale = die_ur - die_ll;
        at::Tensor shift = die_ll;
        st::setting.filler_type = st::setting.use_pre_gp ? "surround" : st::setting.filler_type;
        if (st::setting.filler_type != "center") {
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
            filler_pos = filler_pos_filter + r;
        }
        filler_pos = filler_pos * scale + shift;
        mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
        mov_node_size = torch::cat({mov_node_size, filler_size}, 0);
    }
    if (st::setting.noise_ratio > 0) {
        at::Tensor noise = torch::rand_like(mov_node_pos);
        noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
        mov_node_pos += noise;
    }

    // FIXME: need to expand nor clamp vias?
    // at::Tensor expand_ratio = mov_node_pos.new_ones((mov_node_pos.size(0)));
    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)));
    if (st::setting.clamp_node) {
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        logger.info("unit_len: %f, %f",unit_len[0].item<float>(), unit_len[1].item<float>());
        at::Tensor clamp_mov_node_size = mov_node_size.clamp(unit_len * sqrt(2));
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        /* FIXME: clamp node except for invisible ones */
        for (int i = 0; i != num_nets; i++) {
            if (bonding_map[i].item<int>() == 0) {
                clamp_mov_node_size[i][0] = 0;
                clamp_mov_node_size[i][1] = 0;
            }
        }

        mov_node_size = clamp_mov_node_size;
    }
    node_pos_legal = node_pos.clone();
    return make_tuple(mov_node_pos, mov_node_size, expand_ratio);
}  // END MODULE

//---------------------------------------------------------------------

void ViaData::dump(
    torch::Tensor node_pos, torch::Tensor node_size, torch::Tensor node_die, int cell_mov_lhs, int cell_mov_rhs, torch::Tensor node_orient) {
    // TODO: align to site -> integer
    auto cell_node_pos = node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)});
    auto via_node_pos = node_pos.index({Slice(cell_mov_rhs, None)});
    for (unsigned i = 0; i != num_nets; ++i) {
        // if (bonding_map[i].item<int>() != 0) {
        if (node_die[i + cell_mov_rhs].item<int>() != -1) {
            // FIXME: die space
            via_node_pos[i][0].data() = via_node_pos[i][0].item<int>();
            via_node_pos[i][1].data() = via_node_pos[i][1].item<int>();
        }
    }
    for (int i = 0; i != num_nets; i++) {  // FIXME: num_bond <-> num_nets
        // if (bonding_map[i].item<int>() != 0) {
        if (node_die[i + cell_mov_rhs].item<int>() != -1) {
            rawdb->bondings[i].place(via_node_pos[i][0].item<int>(), via_node_pos[i][1].item<int>());
        } else {
            rawdb->bondings[i].remove();
        }
    }

    /* place rawdb */
    for (int i = 0; i < iopin_mov_lhs; i++) {
        cell_node_pos[i][0].data().copy_(round((cell_node_pos[i][0] - node_size[i][0] / 2).item<float>()) +
                                         node_size[i][0] / 2);
        cell_node_pos[i][1].data().copy_(round((cell_node_pos[i][1] - node_size[i][1] / 2).item<float>()) +
                                         node_size[i][1] / 2);
        rawdb->cells[i]->place_with_orient(round((cell_node_pos[i][0] - node_size[i][0] / 2).item<float>()),
                               round((cell_node_pos[i][1] - node_size[i][1] / 2).item<float>()),
                               node_die[i].item<int>(), node_orient[i].item<int>());
    }
    // cell_node_pos[98113][0].data().copy_(round((cell_node_pos[98113][0] - node_size[98113][0] / 2).item<float>()) +
    //                                      node_size[98113][0] / 2);
    // cell_node_pos[98113][1].data().copy_(round((cell_node_pos[98113][1] - node_size[98113][1] / 2).item<float>()) +
    //                                      node_size[98113][1] / 2);
    
    // cout << cell_node_pos[98113] << endl;

    auto node_pos_dump = torch::cat({cell_node_pos, via_node_pos}, 0);
    node_pos.data().copy_(node_pos_dump.data());
}  // END MODULE

//---------------------------------------------------------------------
