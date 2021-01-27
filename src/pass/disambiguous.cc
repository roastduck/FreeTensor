#include <pass/disambiguous.h>

namespace ir {

Stmt Disambiguous::visitStmt(
    const Stmt &op, const std::function<Stmt(const Stmt &)> &visitNode) {
    if (op->noAmbiguous_) {
        return op;
    }
    auto ret = Mutator::visitStmt(op, visitNode);
    ret->noAmbiguous_ = true;
    return ret;
}

Expr Disambiguous::visitExpr(
    const Expr &op, const std::function<Expr(const Expr &)> &visitNode) {
    if (op->noAmbiguous_) {
        return op;
    }
    auto ret = Mutator::visitExpr(op, visitNode);
    ret->noAmbiguous_ = true;
    return ret;
}

} // namespace ir

