#include <pass/replace_uses.h>

namespace ir {

static Expr makeReduce(ReduceOp reduceOp, const Expr &lhs, const Expr &rhs) {
    switch (reduceOp) {
    case ReduceOp::Add:
        return makeAdd(lhs, rhs);
    case ReduceOp::Mul:
        return makeMul(lhs, rhs);
    case ReduceOp::Max:
        return makeMax(lhs, rhs);
    case ReduceOp::Min:
        return makeMin(lhs, rhs);
    default:
        ASSERT(false);
    }
}

Expr ReplaceUses::visit(const Load &op) {
    if (replaceLoad_.count(op)) {
        return (*this)(replaceLoad_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

Stmt ReplaceUses::visit(const ReduceTo &op) {
    if (replaceReduceTo_.count(op)) {
        return makeStore(
            op->id(), op->var_, op->indices_,
            (*this)(makeReduce(op->op_, op->expr_, replaceReduceTo_.at(op))));
    } else {
        return Mutator::visit(op);
    }
}

} // namespace ir

