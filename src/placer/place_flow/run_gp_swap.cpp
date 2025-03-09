#include "../run_placement.h"

torch::Tensor run_gp_swap(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state,
                          int& cell_mov_lhs, int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs,
                          int& mov_rhs) {
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    auto node_die = data.node_die.clone();
    auto device = data.device;
    /* cell visualization information */
    torch::Tensor node_size_bot = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    torch::Tensor node_size_top = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    for (int i = 0; i != data.num_nodes; i++) {
        if (node_die[i].item<int>() == 0) {
            node_size_bot[i] = data.node_size_bot[i];
        } else {
            node_size_top[i] = data.node_size_top[i];
        }
    }

    /* movable cells */
    torch::Tensor init_density_map = get_init_density_map(data);
    auto [mov_node_pos, mov_node_size, expand_ratio] = data.get_mov_node_info_cross_chip();
    std::tie(mov_lhs, mov_rhs) = data.movable_index;

    torch::Tensor via_init_density_map =
        torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);
    via_data.init_vars();  // TODO: equivalent to PlaceData::init_filler
    auto [via_mov_node_pos, via_mov_node_size, via_expand_ratio] = via_data.get_mov_node_info();
    std::tie(via_mov_lhs, via_mov_rhs) = via_data.movable_index;

    torch::Tensor mov_node_pos_all;
    torch::Tensor mov_node_size_all;
    torch::Tensor expand_ratio_all;

    if (true) {
        auto [via_pos, ovlp_hpwl_info] = wa_wirelength_hpwl::place_via_to_optim(
            node_pos.index({Slice(cell_mov_lhs, cell_mov_rhs)}).to(device), node_die.to(device),
            data.pin_id2node_id.to(device), data.pin_rel_cpos.to(device), data.hyperedge_list.to(device),
            data.hyperedge_list_end.to(device), (via_data.bonding_map == 1).to(device));

        auto ovlp_hpwl = ovlp_hpwl_info.index({0, "..."}).sum();
        auto ovlp_info = ovlp_hpwl_info.index({1, "..."});
        auto num_nets_not_optim = torch::_cast_Int(ovlp_info == 0).sum();
        logger.info("The ovlp wl is %d", ovlp_hpwl.item<int>(), num_nets_not_optim.item<int>());

        std::tie(mov_node_pos_all, mov_node_size_all, expand_ratio_all) = data.get_mov_node_info_cross_chip_with_via(
            via_data, make_tuple(mov_node_pos, mov_node_size, expand_ratio),
            make_tuple(via_mov_node_pos, via_mov_node_size, via_expand_ratio));
        std::tie(mov_lhs, mov_rhs) = data.movable_index;
        logger.info("Concating vias in optim region ...");
        for (int i = 0; i < cell_mov_rhs; i++) {
            mov_node_pos_all[i][0] = node_pos[i][0].item<float>();
            mov_node_pos_all[i][1] = node_pos[i][1].item<float>();
        }
        for (int i = cell_mov_rhs; i < mov_rhs; i++) {
            mov_node_pos_all[i][0] = via_pos[i - data.num_nodes][0].item<float>();
            mov_node_pos_all[i][1] = via_pos[i - data.num_nodes][1].item<float>();
        }
        node_pos = mov_node_pos_all.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);

        auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos.to(device), data.node_die.to(device), data);
        logger.info("Adding vias, exact HPWL of two chips: (bot + top = total): (%d + %d = %d)", hpwl1.item<int>(),
                    hpwl2.item<int>(), (hpwl1 + hpwl2).item<int>());
    }

    printlog(LOG_WARN, "MEM: cur = %.2f MB, peak = %.2f MB", utils::mem_use::get_current(), utils::mem_use::get_peak());

    return node_pos;
}