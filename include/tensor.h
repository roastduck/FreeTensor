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

} // namespace ir

#endif // TENSOR_H
