#include <analyze/normalize.h>

namespace ir {

Stmt Normalize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    op->infoLen_ = makeSub(op->end_, op->begin_);
    return op;
}

Stmt Normalize::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    op->infoNotCond_ = makeLNot(op->cond_);
    return op;
}

} // namespace ir

