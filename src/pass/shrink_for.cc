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
    auto var = makeVar(_op->iter_).as<VarNode>();
    auto hash = getHash(var);
    newRange_.erase(hash);

    iterStack_.emplace_back(var);
    auto __op = SimplifyPass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    iterStack_.pop_back();

    ASSERT(newRange_.count(hash));
    auto newBegin = newRange_.at(hash).first;
    auto newEnd = makeAdd(newRange_.at(hash).second, makeIntConst(1));
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
    auto op = ShrinkFor(keepConst)(_op);
    return simplifyPass(op);
}

} // namespace ir

