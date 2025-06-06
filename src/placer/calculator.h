#pragma once

#include "core/core.h"
#include "database.h"
#include "global.h"
#include "optim/ParamScheduler.h"

torch::Tensor calc_loss(torch::Tensor wl_loss, torch::Tensor density_loss, ParamScheduler& ps);

void apply_precond(torch::Tensor mov_node_pos, ParamScheduler& ps);

tuple<torch::Tensor, torch::Tensor> calc_grad(torch::optim::Optimizer& optimizer,
                                              torch::Tensor mov_node_pos,
                                              torch::Tensor wl_loss,
                                              torch::Tensor density_loss);

tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad(torch::Tensor mov_node_pos,
                                                      std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                      torch::Tensor mov_node_size,
                                                      torch::Tensor init_density_map,
                                                      ElectronicDensityLayer& density_map_layer,
                                                      torch::Tensor conn_fix_node_pos,
                                                      ParamScheduler& ps,
                                                      PlaceData& data);

tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad_with_via(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_map,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data,
    int& c_id);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_optimization(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_map,
    ElectronicDensityLayer& density_map_layer,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    PlaceData& data);

tuple<torch::Tensor, torch::Tensor> calc_obj_and_grad_multi_circuit(
    torch::Tensor mov_node_pos,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor mov_node_size,
    torch::Tensor init_density_map,
    vector<ElectronicDensityLayer>& density_map_layers,
    torch::Tensor conn_fix_node_pos,
    ParamScheduler& ps,
    NodeData& data,
    PlaceData& via_data);