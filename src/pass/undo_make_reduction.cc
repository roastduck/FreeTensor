#include <pass/undo_make_reduction.h>

namespace ir {

Stmt UndoMakeReduction::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    switch (op->op_) {
    case ReduceOp::Add:
        return makeStore(op->id(), op->var_, op->indices_,
                         makeAdd(makeLoad(op->var_, op->indices_), op->expr_));
    case ReduceOp::Mul:
        return makeStore(op->id(), op->var_, op->indices_,
                         makeMul(makeLoad(op->var_, op->indices_), op->expr_));
    case ReduceOp::Min:
        return makeStore(op->id(), op->var_, op->indices_,
                         makeMin(makeLoad(op->var_, op->indices_), op->expr_));
    case ReduceOp::Max:
        return makeStore(op->id(), op->var_, op->indices_,
                         makeMax(makeLoad(op->var_, op->indices_), op->expr_));
    default:
        ASSERT(false);
    }
}

} // namespace ir

