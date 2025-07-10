#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "tensor.h"

namespace utec::neural_network {

    template<typename T> class IOptimizer;

    template<typename T>
    class ILayer {
    public:
        virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;
        virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) = 0;
        virtual void update_params(IOptimizer<T>&) {}
        virtual ~ILayer() = default;
    };

    template<typename T, int Rank>
    class ILoss {
    public:
        virtual T loss() const = 0;
        virtual utec::algebra::Tensor<T, Rank> loss_gradient() const = 0;
        virtual ~ILoss() = default;
    };

    template<typename T>
    class IOptimizer {
    public:
        virtual void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) = 0;
        virtual void step() {}
        virtual ~IOptimizer() = default;
    };

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H 