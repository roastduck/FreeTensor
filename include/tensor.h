#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include <data_type.h>
#include <except.h>
#include <expr.h>

namespace ir {

class Tensor {
    std::vector<SubTree<ExprNode>> shape_;
    DataType dtype_;

  public:
    Tensor(std::vector<SubTree<ExprNode>> &&shape, DataType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {}
    Tensor(const std::vector<SubTree<ExprNode>> &shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}
    Tensor(const std::vector<Expr> &shape, DataType dtype)
        : shape_(shape.begin(), shape.end()), dtype_(dtype) {}
    Tensor(std::initializer_list<Expr> shape, DataType dtype)
        : shape_(shape.begin(), shape.end()), dtype_(dtype) {}

    std::vector<SubTree<ExprNode>> &shape() { return shape_; }
    const std::vector<SubTree<ExprNode>> &shape() const { return shape_; }
    void setShape(std::vector<SubTree<ExprNode>> &&shape) {
        shape_ = std::move(shape);
    }
    void setShape(const std::vector<SubTree<ExprNode>> &shape) {
        shape_ = shape;
    }
    void setShape(const std::vector<Expr> &shape) {
        shape_ = std::vector<SubTree<ExprNode>>(shape.begin(), shape.end());
    }

    DataType dtype() const { return dtype_; }

    bool isScalar() const;
};

} // namespace ir

#endif // TENSOR_H
