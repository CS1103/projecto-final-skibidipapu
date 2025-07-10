#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include <functional>
#include "nn_interfaces.h"
#include "tensor.h"
#include <numeric>
#include <random>

using utec::algebra::Tensor;

namespace utec::neural_network {

    template<typename T>
    class Dense : public ILayer<T> {
        Tensor<T, 2> W_, b_;
        Tensor<T, 2> input_;
        Tensor<T, 2> grad_W_, grad_b_;

    public:
        using InitFunction = std::function<void(Tensor<T, 2>&)>;

        Dense(size_t in_features, size_t out_features,
              InitFunction init_w, InitFunction init_b)
                : W_(in_features, out_features),
                  b_(1, out_features),
                  grad_W_(in_features, out_features),
                  grad_b_(1, out_features) {
            init_w(W_);
            init_b(b_);
        }

        Dense(size_t in_features, size_t out_features)
                : W_(in_features, out_features),
                  b_(1, out_features),
                  grad_W_(in_features, out_features),
                  grad_b_(1, out_features) {
            auto xavier_init = [](Tensor<T, 2>& tensor) {
                std::random_device rd;
                std::mt19937 gen(rd());
                T scale = std::sqrt(T(2.0) / (tensor.shape()[0] + tensor.shape()[1]));
                std::normal_distribution<T> dist(T(0), scale);
                
                for (size_t i = 0; i < tensor.size(); ++i) {
                    tensor[i] = dist(gen);
                }
            };
            
            auto zero_init = [](Tensor<T, 2>& tensor) {
                tensor.fill(T(0));
            };
            
            xavier_init(W_);
            zero_init(b_);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& X) override {
            input_ = X;
            return X.matmul(W_) + b_;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dY) override {
            grad_W_ = input_.transpose().matmul(dY);
            grad_b_ = dY.sum_rows();
            return dY.matmul(W_.transpose());
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W_, grad_W_);
            optimizer.update(b_, grad_b_);
        }
    };
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H 