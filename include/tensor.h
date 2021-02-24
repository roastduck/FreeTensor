#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include <except.h>
#include <expr.h>

namespace ir {

enum class DataType : int { Float32, Int32 };

inline std::string toString(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "f32";
    case DataType::Int32:
        return "i32";
    }
    return "???";
}

inline size_t sizeOf(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
    case DataType::Int32:
        return 4;
    }
    ASSERT(false);
}

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
