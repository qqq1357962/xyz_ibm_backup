
#include "evaluator.h"

namespace GP3D {

tuple<torch::Tensor, torch::Tensor> get_obj_value(torch::Tensor pin_pos, torch::Tensor density_map, PlaceData& data) {
    torch::NoGradGuard no_grad;
    torch::Tensor hpwl = torch::sum(wa_wirelength_hpwl::get_hpwl(data, pin_pos.detach()));
    torch::Tensor overflow_sum = ((density_map - st::setting.target_density) * data.bin_area).clamp_(0.0).sum();
    torch::Tensor overflow = overflow_sum / data.__total_mov_area_without_filler__;
    return {hpwl, overflow};
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_evaluator(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_map,
    ElectronicDensityLayer& density_map_layer,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    PlaceData& data) {
    auto [mov_lhs, mov_rhs] = data.movable_index;

    mov_node_pos = constraint_fn(mov_node_pos);

    auto [overflow, density_map] =
        density_map_layer.direct_calc_overflow(mov_node_pos, mov_node_size, init_density_map);

    auto conn_node_pos = mov_node_pos.index({Slice({mov_lhs, mov_rhs})});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);
    torch::Tensor masked_hpwl = wa_wirelength_hpwl::masked_scale_hpwl(conn_node_pos,
                                                                      data.pin_id2node_id,
                                                                      data.pin_rel_cpos,
                                                                      data.hyperedge_list,
                                                                      data.hyperedge_list_end,
                                                                      data.net_mask,
                                                                      data.hpwl_scale);

    // auto node_die = density_map_layer.node_die.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).clone();
    // node_die = torch::cat({node_die, data.node_die.index({Slice(data.cell_mov_rhs, None)})});
    // torch::Tensor masked_hpwl = wa_wirelength_hpwl::masked_scale_hpwl_cross_chip(conn_node_pos,
    //                                                                              density_map_layer.node_die,
    //                                                                              data.pin_id2node_id,
    //                                                                              data.pin_rel_cpos,
    //                                                                              data.hyperedge_list,
    //                                                                              data.hyperedge_list_end,
    //                                                                              data.net_mask,
    //                                                                              data.hpwl_scale);

    return {masked_hpwl, overflow, density_map};
}  // END MODULE

}  // namespace GP3D