#pragma once

#include "calculator.h"
#include "core/core.h"
#include "database.h"
#include "global.h"
#include "visualization.h"
#include "optim/ParamScheduler.h"

torch::Tensor get_init_density_map(PlaceData& data);

torch::Tensor get_init_density_map_cross_chip(NodeData& data);

void init_params(torch::Tensor mov_node_pos, std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn, int mov_lhs,
                 int mov_rhs, torch::Tensor conn_fix_node_pos, ElectronicDensityLayer& density_map_layer,
                 torch::Tensor mov_node_size, torch::Tensor init_density_map, torch::optim::Optimizer& optimizer,
                 ParamScheduler& ps, PlaceData& data);

double estimate_initial_learning_rate(
    const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn, torch::Tensor _x_k, double lr);

void init_params_with_via(torch::Tensor mov_node_pos, std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn,
                          int mov_lhs, int mov_rhs, torch::Tensor conn_fix_node_pos,
                          vector<ElectronicDensityLayer>& density_map_layers, torch::Tensor mov_node_size,
                          torch::Tensor init_density_map, torch::optim::Optimizer& optimizer, ParamScheduler& ps,
                          NodeData& data, int& c_id);

void init_params_multi_circuit(torch::Tensor mov_node_pos,
                               std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn, int mov_lhs, int mov_rhs,
                               torch::Tensor conn_fix_node_pos, vector<ElectronicDensityLayer>& density_map_layers,
                               torch::Tensor mov_node_size, torch::Tensor init_density_map,
                               torch::optim::Optimizer& optimizer, ParamScheduler& ps, NodeData& data);