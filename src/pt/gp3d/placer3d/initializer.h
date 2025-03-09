#pragma once

#include "../core3d/core.h"
#include "../optim3d/ParamScheduler.h"
#include "calculator.h"
#include "database3d.h"

namespace GP3D {

void init_params(torch::Tensor mov_node_pos,
                 std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn,
                 int mov_lhs,
                 int mov_rhs,
                 torch::Tensor conn_fix_node_pos,
                //  ElectronicDensityLayer& density_map_layer,
                 vector<ElectronicDensityLayer>& density_map_layers,
                 torch::Tensor mov_node_size,
                 torch::Tensor init_density_map,
                 torch::optim::Optimizer& optimizer,
                 ParamScheduler& ps,
                 NodeData3D& data);

double estimate_initial_learning_rate(
    const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn,
    torch::Tensor _x_k,
    double lr);

}  // namespace GP3D