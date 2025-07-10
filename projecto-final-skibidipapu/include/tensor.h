#pragma once
#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace utec {
namespace algebra {

template<typename T, size_t Rank>
class Tensor {
private:
    std::array<size_t, Rank> shape_;
    std::vector<T> data_;

    template<typename... Idxs>
    size_t calculate_index(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == Rank, "Number of indices must match tensor rank");
        std::array<size_t, Rank> indices = {static_cast<size_t>(idxs)...};

        size_t linear_index = 0;
        size_t stride = 1;

        for (int i = Rank - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            linear_index += indices[i] * stride;
            stride *= shape_[i];
        }

        return linear_index;
    }

    size_t total_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
    }

    bool is_broadcast_compatible(const std::array<size_t, Rank>& other_shape) const {
        for (size_t i = 0; i < Rank; ++i) {
            if (shape_[i] != other_shape[i] && shape_[i] != 1 && other_shape[i] != 1) {
                return false;
            }
        }
        return true;
    }

    size_t broadcast_index(size_t linear_idx, const std::array<size_t, Rank>& target_shape) const {
        std::array<size_t, Rank> multi_idx;
        size_t temp = linear_idx;

        for (int i = Rank - 1; i >= 0; --i) {
            multi_idx[i] = temp % target_shape[i];
            temp /= target_shape[i];
        }

        for (size_t i = 0; i < Rank; ++i) {
            if (shape_[i] == 1 && target_shape[i] > 1) {
                multi_idx[i] = 0;
            }
        }

        size_t result = 0;
        size_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            result += multi_idx[i] * stride;
            stride *= shape_[i];
        }

        return result;
    }

public:
    Tensor() : shape_(), data_() {}

    explicit Tensor(const std::array<size_t, Rank>& shape)
        : shape_(shape), data_(total_size()) {}

    template<
        typename... Dims,
        typename = std::enable_if_t<
            sizeof...(Dims) == Rank &&
            (std::conjunction_v<std::is_arithmetic<Dims>...>)
        >
    >
    explicit Tensor(Dims... dims) {
        std::array<size_t, Rank> temp_shape{static_cast<size_t>(dims)...};
        shape_ = temp_shape;
        data_.resize(total_size());
    }

    template<typename... Idxs>
    T& operator()(Idxs... idxs) {
        return data_[calculate_index(idxs...)];
    }

    template<typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        return data_[calculate_index(idxs...)];
    }

    const std::array<size_t, Rank>& shape() const noexcept {
        return shape_;
    }

    template<typename... Dims>
    void reshape(Dims... dims) {
        if (sizeof...(dims) != Rank) {
            throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
        }
        std::array<size_t, Rank> new_shape;
        size_t dims_array[] = {static_cast<size_t>(dims)...};
        std::copy(dims_array, dims_array + Rank, new_shape.begin());

        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());

        if (new_size > data_.size()) {
            throw std::invalid_argument("New shape size cannot be larger than current data size");
        }

        shape_ = new_shape;
        data_.resize(new_size);
    }

    void reshape(const std::array<size_t, Rank>& new_shape) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());

        if (new_size > data_.size()) {
            throw std::invalid_argument("New shape size cannot be larger than current data size");
        }

        shape_ = new_shape;
        data_.resize(new_size);
    }

    void fill(const T& value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    Tensor operator+(const Tensor& other) const {
        if (!is_broadcast_compatible(other.shape_)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        std::array<size_t, Rank> result_shape;
        for (size_t i = 0; i < Rank; ++i) {
            result_shape[i] = std::max(shape_[i], other.shape_[i]);
        }

        Tensor result(result_shape);
        size_t result_size = result.total_size();

        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = broadcast_index(i, result_shape);
            size_t idx2 = other.broadcast_index(i, result_shape);
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }

        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (!is_broadcast_compatible(other.shape_)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        std::array<size_t, Rank> result_shape;
        for (size_t i = 0; i < Rank; ++i) {
            result_shape[i] = std::max(shape_[i], other.shape_[i]);
        }

        Tensor result(result_shape);
        size_t result_size = result.total_size();

        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = broadcast_index(i, result_shape);
            size_t idx2 = other.broadcast_index(i, result_shape);
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }

        return result;
    }

    Tensor operator*(const Tensor& other) const {
        if (!is_broadcast_compatible(other.shape_)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        std::array<size_t, Rank> result_shape;
        for (size_t i = 0; i < Rank; ++i) {
            result_shape[i] = std::max(shape_[i], other.shape_[i]);
        }

        Tensor result(result_shape);
        size_t result_size = result.total_size();

        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = broadcast_index(i, result_shape);
            size_t idx2 = other.broadcast_index(i, result_shape);
            result.data_[i] = data_[idx1] * other.data_[idx2];
        }

        return result;
    }

    Tensor operator+(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val += scalar;
        }
        return result;
    }

    Tensor operator-(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val -= scalar;
        }
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val *= scalar;
        }
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val /= scalar;
        }
        return result;
    }

    Tensor& operator+=(const T& scalar) {
        for (auto& val : data_) {
            val += scalar;
        }
        return *this;
    }

    Tensor transpose_2d() const {
        if constexpr (Rank < 2) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        } else {
            std::array<size_t, Rank> new_shape = shape_;
            std::swap(new_shape[Rank-2], new_shape[Rank-1]);

            Tensor result(new_shape);

            if constexpr (Rank == 2) {
                for (size_t i = 0; i < shape_[0]; ++i) {
                    for (size_t j = 0; j < shape_[1]; ++j) {
                        result(j, i) = (*this)(i, j);
                    }
                }
            } else {
                size_t total_batches = 1;
                for (size_t i = 0; i < Rank - 2; ++i) {
                    total_batches *= shape_[i];
                }

                size_t rows = shape_[Rank-2];
                size_t cols = shape_[Rank-1];

                for (size_t batch = 0; batch < total_batches; ++batch) {
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            size_t src_idx = batch * rows * cols + i * cols + j;
                            size_t dst_idx = batch * rows * cols + j * rows + i;
                            result.data_[dst_idx] = data_[src_idx];
                        }
                    }
                }
            }

            return result;
        }
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    Tensor matmul(const Tensor& other) const {
        return matrix_product(*this, other);
    }

    Tensor transpose() const {
        return transpose_2d();
    }

    Tensor sum_rows() const {
        if constexpr (Rank == 2) {
            Tensor result(1, shape_[1]);
            for (size_t j = 0; j < shape_[1]; ++j) {
                T sum = 0;
                for (size_t i = 0; i < shape_[0]; ++i) {
                    sum += (*this)(i, j);
                }
                result(0, j) = sum;
            }
            return result;
        } else {
            throw std::invalid_argument("sum_rows() only works for 2D tensors");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        if constexpr (Rank == 1) {
            os << "{";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                if (i > 0) os << " ";
                os << tensor.data_[i];
            }
            os << "}";
        } else if constexpr (Rank == 2) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                    if (j > 0) os << " ";
                    os << tensor(i, j);
                }
                os << "\n";
            }
            os << "}";
        } else if constexpr (Rank == 3) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                os << "{\n";
                for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                    for (size_t k = 0; k < tensor.shape_[2]; ++k) {
                        if (k > 0) os << " ";
                        os << tensor(i, j, k);
                    }
                    os << "\n";
                }
                os << "}";
                if (i < tensor.shape_[0] - 1) os << "\n";
            }
            os << "\n}";
        } else {
            os << "{";
            for (size_t i = 0; i < tensor.data_.size(); ++i) {
                if (i > 0) os << " ";
                os << tensor.data_[i];
            }
            os << "}";
        }
        return os;
    }
};

template<typename T, size_t Rank>
Tensor<T, Rank> operator+(const T& scalar, const Tensor<T, Rank>& tensor) {
    return tensor + scalar;
}

template<typename T, size_t Rank>
Tensor<T, Rank> operator*(const T& scalar, const Tensor<T, Rank>& tensor) {
    return tensor * scalar;
}

template<typename T, size_t Rank>
Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& tensor) {
    return tensor.transpose_2d();
}

template<typename T, size_t Rank>
Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
    static_assert(Rank >= 2, "Matrix product requires at least 2D tensors");

    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();

    if (shape_a[Rank-1] != shape_b[Rank-2]) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    if constexpr (Rank > 2) {
        for (size_t i = 0; i < Rank - 2; ++i) {
            if (shape_a[i] != shape_b[i]) {
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }
    }

    std::array<size_t, Rank> result_shape = shape_a;
    result_shape[Rank-1] = shape_b[Rank-1];

    Tensor<T, Rank> result(result_shape);

    if constexpr (Rank == 2) {
        #pragma omp parallel for
        for (size_t i = 0; i < shape_a[0]; ++i) {
            for (size_t j = 0; j < shape_b[1]; ++j) {
                T sum = T{};
                for (size_t k = 0; k < shape_a[1]; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
    } else if constexpr (Rank == 3) {
        for (size_t batch = 0; batch < shape_a[0]; ++batch) {
            for (size_t i = 0; i < shape_a[1]; ++i) {
                for (size_t j = 0; j < shape_b[2]; ++j) {
                    T sum = T{};
                    for (size_t k = 0; k < shape_a[2]; ++k) {
                        sum += a(batch, i, k) * b(batch, k, j);
                    }
                    result(batch, i, j) = sum;
                }
            }
        }
    }

    return result;
}

}
} 