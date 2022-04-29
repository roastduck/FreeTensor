#ifndef FREE_TENSOR_TENSOR_H
#define FREE_TENSOR_TENSOR_H

#include <string>
#include <vector>

#include <data_type.h>
#include <except.h>
#include <expr.h>

namespace freetensor {

class Tensor : public ASTPart {
    template <class T> friend Ref<Tensor> makeTensor(T &&, DataType);
    friend Ref<Tensor> makeTensor(std::initializer_list<Expr>, DataType);

    SubTreeList<ExprNode> shape_ = ChildOf{this};
    DataType dtype_;

  public:
    auto &shape() { return shape_; }
    const auto &shape() const { return shape_; }
    void setShape(SubTreeList<ExprNode> &&shape) { shape_ = std::move(shape); }
    void setShape(const SubTreeList<ExprNode> &shape) { shape_ = shape; }
    void setShape(const std::vector<Expr> &shape) { shape_ = shape; }
    void setShape(std::initializer_list<Expr> shape) { shape_ = shape; }

    DataType dtype() const { return dtype_; }

    bool isScalar() const;

    void compHash() override;
};

template <class T> Ref<Tensor> makeTensor(T &&shape, DataType dtype) {
    auto t = Ref<Tensor>::make();
    t->shape_ = std::forward<T>(shape);
    t->dtype_ = dtype;
    return t;
}
inline Ref<Tensor> makeTensor(std::initializer_list<Expr> shape,
                              DataType dtype) {
    auto t = Ref<Tensor>::make();
    t->shape_ = shape;
    t->dtype_ = dtype;
    return t;
}

inline Ref<Tensor> deepCopy(const Ref<Tensor> &t) {
    return makeTensor(t->shape(), t->dtype());
}

} // namespace freetensor

#endif // FREE_TENSOR_TENSOR_H
