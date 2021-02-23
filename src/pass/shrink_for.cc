#include <pass/disambiguous.h>
#include <pass/shrink_for.h>

namespace ir {

Expr ShrinkFor::simplifyExpr(const Expr &_expr) {
    auto expr = _expr;
    auto hash = getHash(expr);
    for (int i = 0; i < 100; i++) {
        auto newExpr = (*this)(expr);
        auto newHash = getHash(newExpr);
        if (newHash == hash) {
            return expr;
        }
        expr = newExpr, hash = newHash;
    }
    WARNING("ShrinkFor::simplifyExpr not converged");
    return expr;
}

Stmt ShrinkFor::visit(const For &_op) {
    newRange_.erase(_op->iter_);

    auto __op = SimplifyPass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    ASSERT(newRange_.count(op->iter_));
    auto newBegin = newRange_.at(op->iter_).first;
    auto newEnd = makeAdd(newRange_.at(op->iter_).second, makeIntConst(1));
    newBegin = simplifyExpr(newBegin);
    newEnd = simplifyExpr(newEnd);

    if (keepConst_) {
        if (newBegin->nodeType() != ASTNodeType::IntConst ||
            newEnd->nodeType() != ASTNodeType::IntConst) {
            return op;
        }
    }

    op->begin_ = newBegin;
    op->end_ = newEnd;
    return op;
}

Stmt shrinkFor(const Stmt &_op, bool keepConst) {
    auto op = disambiguous(_op); // we are inheriting SimplifyPass
    op = ShrinkFor(keepConst)(op);
    return simplifyPass(op);
}

} // namespace ir

