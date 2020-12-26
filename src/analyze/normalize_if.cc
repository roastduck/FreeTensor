#include <analyze/normalize_if.h>

namespace ir {

Stmt NormalizeIf::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    switch (op->cond_->nodeType()) {
    case ASTNodeType::LT:
        op->info_equival_cond_ =
            makeSub(op->cond_.as<LTNode>()->lhs_, op->cond_.as<LTNode>()->rhs_);
        break;
    case ASTNodeType::GT:
        op->info_equival_cond_ =
            makeSub(op->cond_.as<GTNode>()->rhs_, op->cond_.as<GTNode>()->lhs_);
        break;
        // TODO: LE and GE, but check for int type
    default:;
    }
    return op;
}

} // namespace ir

