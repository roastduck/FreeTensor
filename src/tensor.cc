#include <tensor.h>

namespace ir {

bool Tensor::isScalar() const {
    for (auto &&d : shape_) {
        if (d->nodeType() != ASTNodeType::IntConst) {
            return false;
        } else if (d.as<IntConstNode>()->val_ != 1) {
            return false;
        }
    }
    return true;
}

Ref<Tensor> deepCopy(const Ref<Tensor> &t) {
    std::vector<Expr> shape;
    shape.reserve(t->shape().size());
    for (auto &&dim : t->shape()) {
        shape.emplace_back(deepCopy(dim));
    }
    return Ref<Tensor>::make(std::move(shape), t->dtype());
}

} // namespace ir
