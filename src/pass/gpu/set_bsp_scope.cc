#include <pass/gpu/set_bsp_scope.h>

namespace freetensor {

namespace gpu {

Stmt SetBSPScope::visit(const BSPScope &op) {
    if (inKernel_) {
        throw InvalidProgram("Unexpected nested BSPScope");
    }
    inKernel_ = true;
    auto ret = Mutator::visit(op);
    inKernel_ = false;
    return ret;
}

Stmt SetBSPScope::visit(const For &op) {
    if (!inKernel_ &&
        std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        inKernel_ = true;
        auto ret = makeBSPScope(Mutator::visit(op));
        inKernel_ = false;
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

} // namespace gpu

} // namespace freetensor
