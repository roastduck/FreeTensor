#include <ast.h>
#include <buffer.h>

namespace ir {

Ref<Buffer> deepCopy(const Ref<Buffer> &b) {
    std::vector<Expr> shape;
    shape.reserve(b->tensor().shape().size());
    for (auto &&dim : b->tensor().shape()) {
        shape.emplace_back(deepCopy(dim));
    }
    Tensor t(std::move(shape), b->tensor().dtype());
    return Ref<Buffer>::make(std::move(t), b->atype(), b->mtype());
}

} // namespace ir
