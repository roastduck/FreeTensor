#include <analyze/normalize.h>

namespace ir {

Stmt Normalize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    op->infoLen_ = makeSub(op->end_, op->begin_);
    return op;
}

} // namespace ir

