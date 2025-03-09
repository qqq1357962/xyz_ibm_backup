#pragma once

#include "../core3d/core.h"
#include "../optim3d/ParamScheduler.h"
#include "database3d.h"

namespace GP3D {

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
                                                      //   ElectronicDensityLayer& density_map_layer,
                                                      vector<ElectronicDensityLayer>& density_map_layers,
                                                      torch::Tensor conn_fix_node_pos,
                                                      ParamScheduler& ps,
                                                      NodeData3D& data);
}  // namespace GP3D