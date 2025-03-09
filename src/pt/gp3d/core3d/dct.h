#pragma once

#include "global.h"
namespace GP3D {

class FFTBasisCache {
public:
    FFTBasisCache() { ; }
    unordered_map<int, at::Tensor> dct;
    unordered_map<int, at::Tensor> idct;
};

tuple<at::Tensor, at::Tensor> torch_dct_idct(
    at::Tensor density_map,
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale);

at::Tensor compute_electronic_force(at::Tensor x, at::Tensor scale, at::Tensor coeff, int dim = 0);

at::Tensor dct(at::Tensor x, bool norm = false);

at::Tensor idct(at::Tensor X, bool norm = false);

at::Tensor dct_2d(at::Tensor x, bool norm = false);

at::Tensor idct_2d(at::Tensor X, bool norm = false);

at::Tensor dct_3d(at::Tensor x, bool norm = false);

at::Tensor idct_3d(at::Tensor X, bool norm = false);

}  // namespace GP3D