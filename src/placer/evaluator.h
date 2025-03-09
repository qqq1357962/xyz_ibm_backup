#pragma once

#include "core/core.h"
#include "database.h"
#include "global.h"
#include "optim/ParamScheduler.h"

tuple<torch::Tensor, torch::Tensor> get_obj_value(torch::Tensor pin_pos, torch::Tensor density_map, PlaceData& data);

tuple<torch::Tensor, torch::Tensor> evaluate_placement(torch::Tensor node_pos,
                                                       ElectronicDensityLayer& density_map_layer,
                                                       torch::Tensor init_density_map,
                                                       PlaceData& data);

tuple<torch::Tensor, torch::Tensor> fast_evaluator(torch::Tensor mov_node_pos,
                                                   std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                   torch::Tensor mov_node_size,
                                                   torch::Tensor init_density_map,
                                                   ElectronicDensityLayer& density_map_layer,
                                                   torch::Tensor conn_fix_node_pos,
                                                   ParamScheduler& ps,
                                                   PlaceData& data);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_wl_cross_chip(torch::Tensor node_pos,
                                                           torch::Tensor node_die,
                                                           PlaceData& data);

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> evaluate_wl_ovhd_cross_chip(torch::Tensor node_pos,
                                                                          torch::Tensor node_die, PlaceData& data);

tuple<torch::Tensor, torch::Tensor> fast_evaluator_cross_chip(torch::Tensor mov_node_pos,
                                                              std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                              torch::Tensor mov_node_size,
                                                              torch::Tensor init_density_map,
                                                              vector<ElectronicDensityLayer>& density_map_layers,
                                                              torch::Tensor conn_fix_node_pos,
                                                              ParamScheduler& ps,
                                                              NodeData& data);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_evaluator_multi_circuit(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_map,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data);

tuple<torch::Tensor, torch::Tensor> evaluate_placement_multi_circuit(torch::Tensor node_pos,
                                                                     vector<ElectronicDensityLayer>& density_map_layers,
                                                                     torch::Tensor init_density_maps,
                                                                     NodeData& data);
     
int calc_via_cost(torch::Tensor node_die,PlaceData& data);