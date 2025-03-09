#pragma once
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "global.h"

// namespace torch {
// namespace serialize {
// class OutputArchive;
// class InputArchive;
// }  // namespace serialize
// }  // namespace torch

namespace torch {
namespace optim {
namespace GP3D {
torch::Tensor calc_nesterov_step(int N);

struct TORCH_API NesterovOptions : public OptimizerCloneableOptions<NesterovOptions> {
    NesterovOptions(double lr);
    TORCH_ARG(double, lr);
    TORCH_ARG(int64_t, max_cache_steps) = 10000;
    TORCH_ARG(torch::Tensor, nesterov_step) = calc_nesterov_step(max_cache_steps_);
    TORCH_ARG(float, last_step) = nesterov_step_[max_cache_steps_ - 1].item().toFloat();

public:
    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    TORCH_API friend bool operator==(const NesterovOptions& lhs, const NesterovOptions& rhs);
    ~NesterovOptions() override = default;
    double get_lr() const override;
    void set_lr(const double lr) override;
};

struct TORCH_API NesterovParamState : public OptimizerCloneableParamState<NesterovParamState> {
    TORCH_ARG(int64_t, step) = 0;
    TORCH_ARG(torch::Tensor, u_k);
    TORCH_ARG(torch::Tensor, v_k);
    TORCH_ARG(torch::Tensor, g_k);
    TORCH_ARG(torch::Tensor, obj_k);
    TORCH_ARG(torch::Tensor, alpha_k);
    TORCH_ARG(torch::Tensor, v_kp1);

public:
    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    TORCH_API friend bool operator==(const NesterovParamState& lhs, const NesterovParamState& rhs);
    ~NesterovParamState() override = default;
};

class TORCH_API Nesterov : public Optimizer {
public:
    explicit Nesterov(std::vector<OptimizerParamGroup> param_groups,
                      NesterovOptions defaults,
                      const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn_)
        : Optimizer(std::move(param_groups), std::make_unique<NesterovOptions>(defaults)),
          obj_and_grad_fn(obj_and_grad_fn_) {
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.max_cache_steps() >= 0, "Invalid max_cache_steps value: ", defaults.max_cache_steps());
    }

    explicit Nesterov(std::vector<Tensor> params,
                      // NOLINTNEXTLINE(performance-move-const-arg)
                      NesterovOptions defaults,
                      const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn_)
        : Nesterov({std::move(OptimizerParamGroup(params))}, defaults, obj_and_grad_fn_) {}

    torch::Tensor step(LossClosure closure = nullptr) override;

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

private:
    const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn;

    template <typename Self, typename Archive>
    static void serialize(Self& self, Archive& archive) {
        _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Nesterov);
    }
};
}  // namespace GP3D
}  // namespace optim
}  // namespace torch