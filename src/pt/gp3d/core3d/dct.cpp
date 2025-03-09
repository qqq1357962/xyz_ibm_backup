#include "dct.h"
namespace GP3D {
FFTBasisCache fft_basis_cache;

tuple<at::Tensor, at::Tensor> torch_dct_idct(
    at::Tensor density_map,
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fft_scale) {
    auto [potential_scale,
          potential_coeff,
          force_x_scale,
          force_y_scale,
          force_z_scale,
          force_x_coeff,
          force_y_coeff,
          force_z_coeff] = fft_scale;

    at::Tensor fft_coeff = 4 * dct_3d(density_map);  // FIXME:

    // Real number, M x N
    at::Tensor potential_map = real(idct_3d(fft_coeff * potential_scale)) * potential_coeff;  // M x N

    at::Tensor force_x_map = compute_electronic_force(fft_coeff, force_x_scale, force_x_coeff, 0);
    at::Tensor force_y_map = compute_electronic_force(fft_coeff, force_y_scale, force_y_coeff, 1);
    at::Tensor force_z_map = compute_electronic_force(fft_coeff, force_z_scale, force_z_coeff, 2);

    at::Tensor grad_mat =
        torch::vstack({force_x_map.unsqueeze(0), force_y_map.unsqueeze(0), force_z_map.unsqueeze(0)});  // 3 x M x N x Z

    grad_mat = grad_mat.contiguous();

    return {grad_mat, potential_map};
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor compute_electronic_force(at::Tensor x, at::Tensor scale, at::Tensor coeff, int dim) {
    // E_x: dim == 0, E_y: dim == 1
    assert(x.dim() == 3);
    assert((dim == 0) || (dim == 1) || (dim == 2));
    x = x * scale;

    if (dim == 0)
        x = torch::cat({x.index({Slice(None, 1), "..."}) * 0, x.index({Slice(1, None), "..."}).flip({0})}, 0);
    else if (dim == 1)
        x = torch::cat(
            {x.index({Slice(), Slice(None, 1), Slice()}) * 0, x.index({Slice(), Slice(1, None), Slice()}).flip({1})},
            1);
    else if (dim == 2)
        x = torch::cat({x.index({"...", Slice(None, 1)}) * 0, x.index({"...", Slice(1, None)}).flip({2})}, 2);

    x = real(idct_3d(x));

    x = x * coeff;
    return x;
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor dct(at::Tensor x_, bool norm) {
    auto x_shape = x_.sizes();
    int N = x_shape[x_.dim() - 1];
    torch::Tensor x = x_.contiguous().view({-1, N});

    at::Tensor v =
        torch::cat({x.index({"...", Slice(None, None, 2)}), x.index({"...", Slice(1, None, 2)}).flip({1})}, 1);

    auto Vc = torch::fft::fft(v, c10::nullopt, 1);

    if (fft_basis_cache.dct.find(N) == fft_basis_cache.dct.end()) {
        at::Tensor k = -torch::arange(N, torch::device(x_.device())).index({None, Slice()}) * M_PI / (2 * N);
        fft_basis_cache.dct[N] = torch::complex(torch::cos(k), torch::sin(k));
    }

    auto V = Vc * fft_basis_cache.dct[N];
    if (norm) {
        V.index({"...", 0}) /= sqrt(N) * 2;
        V.index({"...", Slice(1, None)}) /= sqrt(N / 2) * 2;
    }
    V = 2 * real(V).view(x_.sizes());

    return V;
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor idct(at::Tensor X, bool norm) {
    auto x_shape = X.sizes();
    int N = x_shape[X.dim() - 1];
    auto X_v = X.contiguous().view({-1, N}) / 2;

    if (norm) {
        X_v.index({"...", 0}) *= sqrt(N) * 2;
        X_v.index({"...", Slice(1, None)}) *= sqrt(N / 2) * 2;
    }

    if (fft_basis_cache.idct.find(N) == fft_basis_cache.idct.end()) {
        at::Tensor k = torch::arange(N, torch::device(X.device())).index({None, Slice()}) * M_PI / (2 * N);
        fft_basis_cache.idct[N] = torch::complex(torch::cos(k), torch::sin(k));
    }
    auto V_t =
        X_v +
        torch::complex(torch::tensor(0, torch::dtype(torch::kFloat)), torch::tensor(1, torch::dtype(torch::kFloat))) *
            torch::cat({X_v.index({Slice(), Slice(None, 1)}) * 0, -X_v.index({Slice(), Slice(1, None)}).flip({1})}, 1);

    auto V = V_t * fft_basis_cache.idct[N];
    auto v = torch::fft::ifft(V, c10::nullopt, 1);
    auto x = v.new_zeros(v.sizes());

    x.index({"...", Slice(None, None, 2)}) += v.index({"...", Slice(None, N - (N / 2))});
    x.index({"...", Slice(1, None, 2)}) += v.flip({1}).index({"...", Slice(None, N / 2)});

    return x.view(X.sizes());
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor dct_2d(at::Tensor x, bool norm) {
    auto X1 = dct(x, norm);
    auto X2 = dct(X1.transpose(-1, -2), norm);
    return X2.transpose(-1, -2);
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor idct_2d(at::Tensor X, bool norm) {
    auto x1 = idct(X, norm);
    auto x2 = idct(x1.transpose(-1, -2), norm);
    return x2.transpose(-1, -2);
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor dct_3d(at::Tensor x, bool norm) {
    // 3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    // For the meaning of the parameter `norm`, see:
    // https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    // :param x: the input signal
    // :param norm: the normalization, None or 'ortho'
    // :return: the DCT-II of the signal over the last 3 dimensions
    auto X1 = dct(x, norm);
    auto X2 = dct(X1.transpose(-1, -2), norm);
    auto X3 = dct(X2.transpose(-1, -3), norm);
    return X3.transpose(-1, -3).transpose(-1, -2);
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor idct_3d(at::Tensor X, bool norm) {
    // The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    // Our definition of idct is that idct_3d(dct_3d(x)) == x
    // For the meaning of the parameter `norm`, see:
    // https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    // :param X: the input signal
    // :param norm: the normalization, None or 'ortho'
    // :return: the DCT-II of the signal over the last 3 dimensions
    auto x1 = idct(X, norm);
    auto x2 = idct(x1.transpose(-1, -2), norm);
    auto x3 = idct(x2.transpose(-1, -3), norm);
    return x3.transpose(-1, -3).transpose(-1, -2);
}

}  // namespace GP3D