#pragma once

#include "../core3d/core.h"
#include "../optim3d/ParamScheduler.h"
#include "database3d.h"

namespace GP3D {

tuple<torch::Tensor, torch::Tensor> get_obj_value(torch::Tensor pin_pos, torch::Tensor density_map, PlaceData& data);

tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_evaluator(torch::Tensor mov_node_pos,
                                                   std::function<torch::Tensor(torch::Tensor)> constraint_fn,
                                                   torch::Tensor mov_node_size,
                                                   torch::Tensor init_density_map,
                                                   ElectronicDensityLayer& density_map_layer,
                                                   torch::Tensor conn_fix_node_pos,
                                                   ParamScheduler& ps,
                                                   NodeData3D& data,
                                                   torch::Tensor node_slide_state,
                                                   torch::Tensor node_orient_state,
                                                   bool rotate_90);
}  // namespace GP3D