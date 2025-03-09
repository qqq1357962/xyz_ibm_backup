#pragma once

#include <torch/torch.h>

namespace GP3D {
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a tensor on CPU")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

}