#include <pass/gpu/simplex_buffers.h>
#include <pass/shrink_var.h>

namespace ir {

namespace gpu {

void FindSimplexOffset::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        BaseClass::visit(op);
    } else {
        ASSERT(!var2para_.count(op->iter_));
        var2para_[op->iter_] = op->property_.parallel_;
        BaseClass::visit(op);
        var2para_.erase(op->iter_);
    }
}

Stmt ApplySimplexOffset::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        return BaseClass::visit(op);
    } else if (!para2var_.count(op->property_.parallel_)) {
        para2var_[op->property_.parallel_] = op->iter_;
        auto ret = BaseClass::visit(op);
        para2var_.erase(op->property_.parallel_);
        return ret;
    } else {
        auto oldVar = para2var_[op->property_.parallel_];
        para2var_[op->property_.parallel_] = op->iter_;
        auto ret = BaseClass::visit(op);
        para2var_[op->property_.parallel_] = oldVar;
        return ret;
    }
}

Stmt simplexBuffers(const Stmt &_op) {
    FindSimplexOffset visitor;
    visitor(_op);
    ApplySimplexOffset mutator(visitor.offsets());
    auto op = mutator(_op);
    return shrinkVar(op);
}

} // namespace gpu

} // namespace ir
