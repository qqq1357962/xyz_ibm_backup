#include "Nesterov.h"

#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <functional>

namespace torch {
namespace optim {
namespace GP3D {
torch::Tensor calc_nesterov_step(int N) {
    torch::Tensor a_k = torch::empty(N + 1, torch::TensorOptions().dtype(torch::kFloat32));
    a_k[0] = 1.0;
    for (int i = 1; i < N + 1; i++) {
        a_k[i] = (1 + torch::sqrt(4 * torch::pow(a_k[i - 1], 2) + 1)) / 2;
    }
    torch::Tensor a_kp1 = torch::roll(a_k, -1).index({torch::indexing::Slice(0, N)});
    return (a_k.index({torch::indexing::Slice(0, N)}) - 1) / a_kp1;
}

// ============ NesterovOptions =============
NesterovOptions::NesterovOptions(double lr) : lr_(lr) {}

bool operator==(const NesterovOptions& lhs, const NesterovOptions& rhs) {
    return (lhs.lr() == rhs.lr()) && (lhs.max_cache_steps() == rhs.max_cache_steps()) &&
           torch::equal(lhs.nesterov_step(), rhs.nesterov_step()) && (lhs.last_step() == rhs.last_step());
}

void NesterovOptions::serialize(torch::serialize::OutputArchive& archive) const {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_cache_steps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(nesterov_step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(last_step);
}

void NesterovOptions::serialize(torch::serialize::InputArchive& archive) {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, max_cache_steps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, nesterov_step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(float, last_step);
}

double NesterovOptions::get_lr() const { return lr(); }

void NesterovOptions::set_lr(const double lr) { this->lr(lr); }

// ============ NesterovParamState =============
bool operator==(const NesterovParamState& lhs, const NesterovParamState& rhs) {
    return (lhs.step() == rhs.step()) && torch::equal(lhs.u_k(), rhs.u_k()) && torch::equal(lhs.v_k(), rhs.v_k()) &&
           torch::equal(lhs.g_k(), rhs.g_k()) && torch::equal(lhs.obj_k(), rhs.obj_k()) &&
           torch::equal(lhs.alpha_k(), rhs.alpha_k()) && torch::equal(lhs.v_kp1(), rhs.v_kp1());
}

void NesterovParamState::serialize(torch::serialize::OutputArchive& archive) const {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(u_k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(v_k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(g_k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(obj_k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha_k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(v_kp1);
}

void NesterovParamState::serialize(torch::serialize::InputArchive& archive) {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, u_k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, v_k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, g_k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, obj_k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, alpha_k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, v_kp1);
}

Tensor Nesterov::step(LossClosure closure) {
    // NoGradGuard no_grad;

    Tensor loss = {};
    if (closure != nullptr) {
        TORCH_WARN("We don't use the closure function in Nesterov anymore");
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }
    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) {
                continue;
            }
            auto grad = p.grad();
            // TORCH_CHECK(!grad.is_sparse(), "Nesterov does not support sparse gradients");
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto& options = static_cast<NesterovOptions&>(group.options());
            bool need_init_alpha_k = false;
            // State initialization
            if (param_state == state_.end()) {
                auto state = std::make_unique<NesterovParamState>();
                state->step(0);
                // u_k is major solution
                state->u_k(p.data().clone(MemoryFormat::Preserve));
                // v_k is reference solution
                state->v_k(p.data().clone(MemoryFormat::Preserve));
                auto [my_obj, my_grad] = this->obj_and_grad_fn(p);
                // # g_k is gradient to v_k
                state->g_k(my_grad.data().clone(MemoryFormat::Preserve));
                // obj_k is the objective at v_k
                state->obj_k(my_obj.data().clone(MemoryFormat::Preserve));
                state->alpha_k(torch::tensor(
                    0.0,
                    torch::TensorOptions().dtype(p.dtype()).device(p.device()).memory_format(MemoryFormat::Preserve)));
                state->v_kp1(torch::zeros_like(p, MemoryFormat::Preserve).requires_grad_(true));
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
                need_init_alpha_k = true;
            }

            auto& state = static_cast<NesterovParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            int64_t step = state.step();
            auto& u_k = state.u_k();
            auto& v_k = state.v_k();
            auto& g_k = state.g_k();
            auto& obj_k = state.obj_k();
            auto& alpha_k = state.alpha_k();
            auto& v_kp1 = state.v_kp1();

            if (need_init_alpha_k) {
                need_init_alpha_k = false;
                Tensor v_k_1 = (v_k - options.lr() * g_k).detach().requires_grad_(true);
                auto [obj_k_1, g_k_1] = this->obj_and_grad_fn(v_k_1);
                alpha_k.data().copy_((v_k - v_k_1).norm(2) / (g_k - g_k_1).norm(2));
            }

            float step_coef = options.last_step();
            if (step < options.max_cache_steps()) {
                step_coef = options.nesterov_step()[step].item().toFloat();
            }

            int backtrack_cnt = 0;
            int max_backtrack_cnt = 10;
            torch::Tensor u_kp1, f_kp1, g_kp1, alpha_kp1;

            while (true) {
                u_kp1 = v_k - alpha_k * g_k;
                v_kp1.data().copy_(u_kp1 + step_coef * (u_kp1 - u_k));

                auto results = this->obj_and_grad_fn(v_kp1);
                f_kp1 = std::get<0>(results);
                g_kp1 = std::get<1>(results);

                alpha_kp1 = (v_kp1.data() - v_k.data()).norm(2) / (g_kp1.data() - g_k.data()).norm(2);
                backtrack_cnt += 1;

                if (alpha_kp1.item().toFloat() > 0.95 * alpha_k.item().toFloat() ||
                    backtrack_cnt >= max_backtrack_cnt) {
                    break;
                }
                alpha_k.data().copy_(alpha_kp1.data());
            }
            loss = obj_k.data().clone();

            u_k.data().copy_(u_kp1.data());
            v_k.data().copy_(v_kp1.data());
            g_k.data().copy_(g_kp1.data());
            obj_k.data().copy_(f_kp1.data());
            alpha_k.data().copy_(alpha_kp1.data());

            p.data().copy_(v_k.data());
            state.step(state.step() + 1);
        }
    }
    return loss;
}

void Nesterov::save(torch::serialize::OutputArchive& archive) const { serialize(*this, archive); }

void Nesterov::load(torch::serialize::InputArchive& archive) {
    IValue pytorch_version;
    if (archive.try_read("pytorch_version", pytorch_version)) {
        serialize(*this, archive);
    } else {  // deserializing archives saved in old format (prior to version 1.5.0)
        logger.error(
            "Your serialized Nesterov optimizer is still using the old serialization format. "
            "You should re-save your Nesterov optimizer to use the new serialization format.");
    }
}
}  // namespace GP3D
}  // namespace optim
}  // namespace torch