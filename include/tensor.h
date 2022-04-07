#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include <data_type.h>
#include <except.h>
#include <expr.h>

namespace ir {

class Tensor : public ASTPart {
    SubTreeList<ExprNode> shape_;
    DataType dtype_;

  public:
    Tensor(SubTreeList<ExprNode> &&shape, DataType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {}
    Tensor(const SubTreeList<ExprNode> &shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}
    Tensor(const std::vector<Expr> &shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}
    Tensor(std::initializer_list<Expr> shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}

    auto &shape() { return shape_; }
    const auto &shape() const { return shape_; }
    void setShape(SubTreeList<ExprNode> &&shape) { shape_ = std::move(shape); }
    void setShape(const SubTreeList<ExprNode> &shape) { shape_ = shape; }
    void setShape(const std::vector<Expr> &shape) { shape_ = shape; }
    void setShape(std::initializer_list<Expr> shape) { shape_ = shape; }

    DataType dtype() const { return dtype_; }

    bool isScalar() const;
};

Ref<Tensor> deepCopy(const Ref<Tensor> &t);

} // namespace ir

#endif // TENSOR_H
