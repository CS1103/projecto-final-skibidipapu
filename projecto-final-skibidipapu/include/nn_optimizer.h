#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

using utec::algebra::Tensor;

namespace utec::neural_network {

    template<typename T>
    class SGD final : public IOptimizer<T> {
        T lr_;
    public:
        explicit SGD(T lr = 0.01) : lr_{lr} {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            for (size_t i = 0; i < params.size(); ++i)
                params[i] -= lr_ * grads[i];
        }
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
        T lr_, beta1_, beta2_, eps_;
        size_t t_ = 0;
        Tensor<T, 2> m_, v_;

    public:
        Adam(T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8)
                : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            if (m_.empty()) {
                m_ = Tensor<T, 2>(param.shape()[0], param.shape()[1]);
                m_.fill(T(0));
            }
            if (v_.empty()) {
                v_ = Tensor<T, 2>(param.shape()[0], param.shape()[1]);
                v_.fill(T(0));
            }

            ++t_;
            for (size_t i = 0; i < param.size(); ++i) {
                m_[i] = beta1_ * m_[i] + (1 - beta1_) * grad[i];
                v_[i] = beta2_ * v_[i] + (1 - beta2_) * grad[i] * grad[i];
                T m_hat = m_[i] / (1 - std::pow(beta1_, t_));
                T v_hat = v_[i] / (1 - std::pow(beta2_, t_));
                param[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    };

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H 