#pragma once
#include <torch/torch.h>

void showtensor(torch::Tensor t);

void showtensor(torch::Tensor t, int from, int to);