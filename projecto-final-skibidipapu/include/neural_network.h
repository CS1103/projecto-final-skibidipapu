#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include "nn_activation.h"
#include "nn_dense.h"
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.push_back(std::move(layer));
        }

        template<template<typename...> class LossType = BCELoss, template<typename...> class OptimizerType = SGD>
        void train(const utec::algebra::Tensor<T,2>& X, const utec::algebra::Tensor<T,2>& Y,
                   const size_t epochs, const size_t batch_size, T lr) {
            OptimizerType<T> optimizer(lr);
            size_t num_samples = X.shape()[0];
            size_t num_batches = (num_samples + batch_size - 1) / batch_size;

            auto last = std::chrono::high_resolution_clock::now();
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                T epoch_loss = 0;
                for (size_t batch = 0; batch < num_batches; ++batch) {
                    size_t start = batch * batch_size;
                    size_t end = std::min(start + batch_size, num_samples);
                    size_t current_batch_size = end - start;

                    // Crear batch de entrada y salida
                    utec::algebra::Tensor<T,2> x_batch(current_batch_size, X.shape()[1]);
                    utec::algebra::Tensor<T,2> y_batch(current_batch_size, Y.shape()[1]);
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        for (size_t j = 0; j < X.shape()[1]; ++j)
                            x_batch(i, j) = X(start + i, j);
                        for (size_t j = 0; j < Y.shape()[1]; ++j)
                            y_batch(i, j) = Y(start + i, j);
                    }

                    auto out = x_batch;
                    for (auto& layer : layers_)
                        out = layer->forward(out);

                    LossType<T> loss(out, y_batch);
                    epoch_loss += loss.loss();
                    auto grad = loss.loss_gradient();

                    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                        grad = (*it)->backward(grad);

                    for (auto& layer : layers_)
                        layer->update_params(optimizer);
                }
                // Mostrar progreso cada 50 épocas
                if ((epoch + 1) % 50 == 0 || epoch == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - last).count();
                    std::cout << "Epoca " << (epoch + 1) << " - Loss promedio: " << (epoch_loss / num_batches)
                              << " - Tiempo desde el ultimo avance: " << elapsed << " s\n" << std::flush;
                    last = now;
                }
            }
        }

        utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
            auto output = X;
            for (auto& layer : layers_)
                output = layer->forward(output);
            return output;
        }

        void add_dense_layer(size_t in_features, size_t out_features) {
            add_layer(std::make_unique<Dense<T>>(in_features, out_features));
        }

        void add_relu_layer() {
            add_layer(std::make_unique<ReLU<T>>());
        }

        void add_sigmoid_layer() {
            add_layer(std::make_unique<Sigmoid<T>>());
        }

        void add_softmax_layer() {
            add_layer(std::make_unique<Softmax<T>>());
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H 