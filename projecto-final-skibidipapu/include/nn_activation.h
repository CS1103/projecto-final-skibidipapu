#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> z_;
    public:
        ReLU() = default;
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
            z_ = z;
            utec::algebra::Tensor<T, 2> result = z;
            for (size_t i = 0; i < result.size(); ++i)
                result[i] = std::max(T(0), z[i]);
            return result;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            utec::algebra::Tensor<T, 2> dz = grad;
            for (size_t i = 0; i < dz.size(); ++i)
                dz[i] = (z_[i] > 0) ? grad[i] : T(0);
            return dz;
        }
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
        utec::algebra::Tensor<T, 2> input_;

    public:
        Sigmoid() = default;
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) override {
            input_ = input;
            auto output = input;
            for (size_t i = 0; i < output.size(); ++i)
                output[i] = T(1) / (T(1) + std::exp(-input[i]));
            return output;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            auto grad_output = grad;
            for (size_t i = 0; i < input_.size(); ++i) {
                T sig = T(1) / (T(1) + std::exp(-input_[i]));
                grad_output[i] = grad[i] * sig * (T(1) - sig);
            }
            return grad_output;
        }
    };

    template<typename T>
    class Softmax final : public ILayer<T> {
        utec::algebra::Tensor<T, 2> input_;

    public:
        Softmax() = default;
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) override {
            input_ = input;
            auto output = input;
            
            size_t num_samples = input.shape()[0];
            size_t num_classes = input.shape()[1];
            
            for (size_t sample = 0; sample < num_samples; ++sample) {
                T max_val = input(sample, 0);
                for (size_t j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, input(sample, j));
                }
                
                T sum = T(0);
                for (size_t j = 0; j < num_classes; ++j) {
                    output(sample, j) = std::exp(input(sample, j) - max_val);
                    sum += output(sample, j);
                }
                
                for (size_t j = 0; j < num_classes; ++j) {
                    output(sample, j) /= sum;
                }
            }
            
            return output;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            return grad;
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H 