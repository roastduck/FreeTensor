#include <pass/disambiguous.h>
#include <pass/use_builtin_div.h>

namespace ir {

static Expr makeNeg(const Expr &expr) { return makeSub(makeIntConst(0), expr); }

Expr UseBuiltinDiv::visit(const FloorDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (getIntLower(op->lhs_) >= 0 && getIntLower(op->rhs_) >= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (getIntLower(op->lhs_) >= 0 && getIntUpper(op->rhs_) <= 0) {
        return makeNeg(makeRoundTowards0Div(op->lhs_, makeNeg(op->rhs_)));
    }
    if (getIntUpper(op->lhs_) <= 0 && getIntLower(op->lhs_) >= 0) {
        return makeNeg(makeRoundTowards0Div(makeNeg(op->lhs_), op->rhs_));
    }
    if (getIntUpper(op->lhs_) <= 0 && getIntUpper(op->rhs_) <= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    return op;
}

Expr UseBuiltinDiv::visit(const CeilDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (getIntLower(op->lhs_) >= 0 && getIntLower(op->rhs_) >= 0) {
        // In case of unsigned data types
        return makeRoundTowards0Div(
            makeAdd(op->lhs_, makeSub(op->rhs_, makeIntConst(1))), op->rhs_);
    }
    if (getIntLower(op->lhs_) >= 0 && getIntUpper(op->rhs_) <= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (getIntUpper(op->lhs_) <= 0 && getIntLower(op->lhs_) >= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (getIntUpper(op->lhs_) <= 0 && getIntUpper(op->rhs_) <= 0) {
        return makeNeg(makeRoundTowards0Div(op->lhs_, makeNeg(op->rhs_)));
    }
    return op;
}

Stmt useBuiltinDiv(const Stmt &_op) {
    auto op = disambiguous(_op);
    op = UseBuiltinDiv()(op);
    return op;
}

} // namespace ir

