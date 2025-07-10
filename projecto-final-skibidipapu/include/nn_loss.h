#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
        utec::algebra::Tensor<T, 2> y_pred_, y_true_;
    public:
        MSELoss(const utec::algebra::Tensor<T, 2> &y_pred, const utec::algebra::Tensor<T, 2> &y_true)
                : y_pred_{y_pred}, y_true_{y_true} {}

        T loss() const override {
            T sum = 0;
            for (size_t i = 0; i < y_pred_.size(); ++i) {
                T diff = y_pred_[i] - y_true_[i];
                sum += diff * diff;
            }
            return sum / y_pred_.size();
        }

        utec::algebra::Tensor<T, 2> loss_gradient() const override {
            utec::algebra::Tensor<T, 2> grad = y_pred_;
            for (size_t i = 0; i < grad.size(); ++i)
                grad[i] = 2 * (y_pred_[i] - y_true_[i]) / grad.size();
            return grad;
        }
    };

    template<typename T>
    class BCELoss final : public ILoss<T, 2> {
        utec::algebra::Tensor<T, 2> y_pred_, y_true_;
    public:
        BCELoss(const  utec::algebra::Tensor<T, 2> &y_pred, const  utec::algebra::Tensor<T, 2> &y_true)
                : y_pred_{y_pred}, y_true_{y_true} {}

        T loss() const override {
            T sum = 0;
            for (size_t i = 0; i < y_pred_.size(); ++i) {
                T yp = std::min(std::max(y_pred_[i], T(1e-7)), T(1 - 1e-7));
                sum += -y_true_[i] * std::log(yp) - (1 - y_true_[i]) * std::log(1 - yp);
            }
            return sum / y_pred_.size();
        }

        utec::algebra::Tensor<T, 2> loss_gradient() const override {
            utec::algebra::Tensor<T, 2> grad = y_pred_;
            for (size_t i = 0; i < grad.size(); ++i) {
                T yp = std::min(std::max(y_pred_[i], T(1e-7)), T(1 - 1e-7));
                grad[i] = (yp - y_true_[i]) / (yp * (1 - yp) * grad.size());
            }
            return grad;
        }
    };

    template<typename T>
    class CrossEntropyLoss final : public ILoss<T, 2> {
        utec::algebra::Tensor<T, 2> y_pred_, y_true_;
    public:
        CrossEntropyLoss(const utec::algebra::Tensor<T, 2>& y_pred, const utec::algebra::Tensor<T, 2>& y_true)
                : y_pred_{y_pred}, y_true_{y_true} {}

        T loss() const override {
            T sum = 0;
            size_t num_samples = y_pred_.shape()[0];
            size_t num_classes = y_pred_.shape()[1];
            
            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t j = 0; j < num_classes; ++j) {
                    if (y_true_(i, j) > 0) {
                        T yp = std::min(std::max(y_pred_(i, j), T(1e-15)), T(1 - 1e-15));
                        sum += -y_true_(i, j) * std::log(yp);
                    }
                }
            }
            return sum / num_samples;
        }

        utec::algebra::Tensor<T, 2> loss_gradient() const override {
            utec::algebra::Tensor<T, 2> grad = y_pred_;
            size_t num_samples = y_pred_.shape()[0];
            size_t num_classes = y_pred_.shape()[1];
            
            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t j = 0; j < num_classes; ++j) {
                    T yp = std::min(std::max(y_pred_(i, j), T(1e-15)), T(1 - 1e-15));
                    grad(i, j) = (yp - y_true_(i, j)) / num_samples;
                }
            }
            return grad;
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H 