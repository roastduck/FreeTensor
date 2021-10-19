#include <pass/gpu/simplex_buffers.h>
#include <pass/shrink_var.h>

namespace ir {

namespace gpu {

void FindSimplexOffset::visit(const VarDef &op) {
    ASSERT(!defs_.count(op->name_));
    defs_[op->name_] = op;
    Visitor::visit(op);
    defs_.erase(op->name_);
}

void FindSimplexOffset::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        Visitor::visit(op);
    } else {
        ASSERT(!var2para_.count(op->iter_));
        var2para_[op->iter_] = op->property_.parallel_;
        Visitor::visit(op);
        var2para_.erase(op->iter_);
    }
}

Stmt ApplySimplexOffset::visit(const VarDef &op) {
    ASSERT(!defs_.count(op->name_));
    defs_[op->name_] = op;
    auto ret = Mutator::visit(op);
    defs_.erase(op->name_);
    return ret;
}

Stmt ApplySimplexOffset::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        return Mutator::visit(op);
    } else if (!para2var_.count(op->property_.parallel_)) {
        para2var_[op->property_.parallel_] = op->iter_;
        auto ret = Mutator::visit(op);
        para2var_.erase(op->property_.parallel_);
        return ret;
    } else {
        auto oldVar = para2var_[op->property_.parallel_];
        para2var_[op->property_.parallel_] = op->iter_;
        auto ret = Mutator::visit(op);
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
