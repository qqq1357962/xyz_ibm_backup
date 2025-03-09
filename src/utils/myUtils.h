#pragma once
#include <torch/torch.h>
#include "placer/database.h"

torch::Tensor load_from_file(std::string path, NodeData& data);
