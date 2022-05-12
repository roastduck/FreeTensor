#include <pass/replace_uses.h>

namespace freetensor {

Expr ReplaceUses::visit(const Load &op) {
    if (replace_.count(op)) {
        return (*this)(replace_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

Stmt ReplaceUses::visit(const ReduceTo &op) {
    if (replace_.count(op)) {
        switch (op->op_) {
        case ReduceOp::Add:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeAdd(replace_.at(op), op->expr_)));
        case ReduceOp::Mul:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeMul(replace_.at(op), op->expr_)));
        case ReduceOp::Min:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeMin(replace_.at(op), op->expr_)));
        case ReduceOp::Max:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeMax(replace_.at(op), op->expr_)));
        case ReduceOp::LAnd:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeLAnd(replace_.at(op), op->expr_)));
        case ReduceOp::LOr:
            return (*this)(makeStore(op->id(), op->var_, op->indices_,
                                     makeLOr(replace_.at(op), op->expr_)));
        default:
            ASSERT(false);
        }
    } else {
        return Mutator::visit(op);
    }
}

} // namespace freetensor
