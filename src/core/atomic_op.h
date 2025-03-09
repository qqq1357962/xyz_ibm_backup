#pragma once

#include <omp.h>

#include <limits>

#include "global.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a tensor on CPU")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// #define AT_TENSOR_SCALARTYPE(TENSOR) TENSOR.scalar_type()
// #define AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
//   AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, __VA_ARGS__)

// #define AT_DISPATCH_FLOATING_TYPES(TENSOR, NAME, ...)                         \
//   [&] {                                                                       \
//     at::ScalarType _st = AT_TENSOR_SCALARTYPE(TENSOR);                        \
//     switch (_st) {                                                            \
//       AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
//       AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
//       default:                                                                \
//         AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        \
//     }                                                                         \
//   }()

template <typename T, bool = std::is_integral<T>::value>
struct AtomicAdd {
    typedef T type;

    /// @brief constructor
    AtomicAdd(type = 1) {}
    template <typename V>
    inline void operator()(type* dst, V v) const {
#pragma omp atomic
        *dst += v;
    }
};
