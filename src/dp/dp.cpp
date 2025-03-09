#include "dp.h"

namespace dp {

void legalizationV1(
    NodeData& data, torch::Tensor node_pos_lg, torch::Tensor node_size_lg, int cell_mov_lhs, int cell_mov_rhs) {
    /* Greedy Legalization */
    auto node_pos_lg_init = node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)}).clone();
    for (int i = 0; i < 2; i++) {
        torch::Tensor node_weight = data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)});
        greedyLegalization(node_pos_lg_init,
                           node_size_lg,
                           node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)}),
                           node_weight,
                           data.die_info,
                           data.numRows[i],
                           data.rowHeights[i],
                           1,
                           64,
                           cell_mov_rhs - cell_mov_lhs);
    }
    /* Abacus Legalization */
    for (int i = 0; i < 2; i++) {
        torch::Tensor node_weight = data.mov_node_weights[i].index({Slice(cell_mov_lhs, cell_mov_rhs)});
        abacusLegalization(node_pos_lg_init,
                           node_size_lg,
                           node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)}),
                           node_weight,
                           data.core_info,
                           data.numRows[i],
                           data.rowHeights[i],
                           1,
                           64,
                           cell_mov_rhs - cell_mov_lhs);
    }
}

void legalizationV2(NodeData& data,
                    DetailedPlaceDataTensor& lg_db_at,
                    torch::Tensor node_pos_lg,
                    int num_sites_y,
                    float row_height,
                    float row_start,
                    int num_bins_x,
                    int num_bins_y,
                    int cell_mov_lhs, 
                    int cell_mov_rhs,
                    bool only_macro) {
    logger.info("row_height: %f", row_height);
    DetailedPlaceData dp_db(data, lg_db_at, num_sites_y, row_height);
    dp_db.yl = max(dp_db.yl , row_start);
    dp_db.yh = min(dp_db.yh , row_start + row_height * num_sites_y);
    bool legal = false;
    
    auto node_die = data.node_die.clone();
    torch::Tensor node_size_bot =data.node_size_bot * (1 - node_die).index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
    torch::Tensor node_size_top = data.node_size_top * node_die.index({Slice(cell_mov_lhs, cell_mov_rhs)}).unsqueeze(1);
    
    auto cell_node_pos_lg = node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)});
    auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_check_top_0");
    auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_check_bottom_0");
    int cnt=0;
    draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info2);
    draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info1);
    
    int count = 0;
    while(!legal) {
        legal = macroLegalization(dp_db, num_bins_x, num_bins_y);
        count++;
        if(count >= 5) {
            exit(0);
        }
        
        cnt++;
        info1 = make_tuple(st::setting.round_recursion, cnt, data.design_name + "_check_top_0");
        info2 = make_tuple(st::setting.round_recursion, cnt, data.design_name + "_check_bottom_0");
        node_pos_lg.index({"...", 0}).data().copy_(lg_db_at.x + lg_db_at.node_size_x / 2);
        node_pos_lg.index({"...", 1}).data().copy_(lg_db_at.y + lg_db_at.node_size_y / 2);
        auto cell_node_pos_lg = node_pos_lg.index({Slice(cell_mov_lhs, cell_mov_rhs)});
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_top, data, info1);
        draw_fig_with_cairo_cpp(cell_node_pos_lg, node_size_bot, data, info2);
        
    }
    if(!only_macro)
    {
        greedyLegalizationV2(dp_db, num_bins_x, num_bins_y);
        abacusLegalizationV2(dp_db, num_bins_x, num_bins_y);
    }else{
        logger.info("only legalize macros, skip std cell legalization!");
    }
    
    lg_db_at.update_node_pos(node_pos_lg);//db.x,db.y-->node_pos_lg,
}

void detail_placement(NodeData& data,
                      DetailedPlaceDataTensor& dp_db_at,
                      torch::Tensor node_pos_dp,
                      int num_sites_y,
                      float row_height,
                      int num_bins_x,
                      int num_bins_y,
                      bool via_dp) {
    DetailedPlaceData dp_db(data, dp_db_at, num_sites_y, row_height);
    dp_db.via_dp = via_dp;
    dp_db.i_bgn = via_dp ? dp_db.num_movable_nodes : 0;
    dp_db.i_end = via_dp ? dp_db.num_nodes : dp_db.num_movable_nodes;

    legalityCheck_main(dp_db);

    kReorder(dp_db, num_bins_x, num_bins_y);
    dp_db_at.update_node_pos(node_pos_dp);
    auto [hpwl1, hpwl2, tmp] = evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
    logger.info("After 1st K-Reorder, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                hpwl1.item<float>(),
                hpwl2.item<float>(),
                (hpwl1 + hpwl2).item<float>());

    legalityCheck_main(dp_db);

    independentSetMatching(dp_db, num_bins_x, num_bins_y);
    dp_db_at.update_node_pos(node_pos_dp);
    std::tie(hpwl1, hpwl2, tmp) = evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
    logger.info("After Independent Set Matching, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                hpwl1.item<float>(),
                hpwl2.item<float>(),
                (hpwl1 + hpwl2).item<float>());

    legalityCheck_main(dp_db);

    globalSwap(dp_db, num_bins_x / 2, num_bins_y / 2, 2, 32);
    dp_db_at.update_node_pos(node_pos_dp);
    std::tie(hpwl1, hpwl2, tmp) = evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
    logger.info("After Global Swap, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                hpwl1.item<float>(),
                hpwl2.item<float>(),
                (hpwl1 + hpwl2).item<float>());

    legalityCheck_main(dp_db);

    kReorder(dp_db, num_bins_x, num_bins_y);
    dp_db_at.update_node_pos(node_pos_dp);
    std::tie(hpwl1, hpwl2, tmp) = evaluate_wl_cross_chip(node_pos_dp.to(data.device), data.node_die.to(data.device), data);
    logger.info("After 2nd K-Reorder, solution eval, exact HPWL (bot, top, total): (%.2f, %.2f, %.2f)",
                hpwl1.item<float>(),
                hpwl2.item<float>(),
                (hpwl1 + hpwl2).item<float>());
}

}  // namespace dp
