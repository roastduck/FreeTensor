#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include <data_type.h>
#include <except.h>
#include <expr.h>

namespace ir {

class Tensor {
    std::vector<Expr> shape_;
    DataType dtype_;

  public:
    Tensor(std::vector<Expr> &&shape, DataType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {}
    Tensor(const std::vector<Expr> &shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}

    std::vector<Expr> &shape() { return shape_; }
    const std::vector<Expr> &shape() const { return shape_; }
    template <class T> void setShape(T &&shape) {
        shape_ = std::forward<T>(shape);
    }

    DataType dtype() const { return dtype_; }
};

class TensorData {
    int size_;
    std::vector<int> shape_;
    std::vector<Expr> data_;

  public:
    TensorData(const std::vector<int> &shape, const std::vector<Expr> &data);
    TensorData(const std::vector<int> &shape, std::vector<Expr> &&data);

    TensorData(const std::vector<int> &shape, const std::vector<int> &data);
    TensorData(const std::vector<int> &shape, const std::vector<double> &data);

    int ndim() const { return shape_.size(); }
    int size() const { return size_; }

    std::vector<int> indices(int offset) const;

    const Expr &at(int offset) const;
};

} // namespace ir

#endif // TENSOR_H
