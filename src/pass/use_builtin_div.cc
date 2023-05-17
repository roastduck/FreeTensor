#include <pass/use_builtin_div.h>

namespace freetensor {

static Expr makeNeg(const Expr &expr) { return makeSub(makeIntConst(0), expr); }

Stmt UseBuiltinDiv::visitStmt(const Stmt &op) {
    auto boundOfOuterStmt = bound_;
    bound_ = Ref<CompUniqueBounds>::make(*this);
    auto ret = BaseClass::visitStmt(op);
    bound_ = boundOfOuterStmt;
    return ret;
}

Expr UseBuiltinDiv::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    // Get bounds of children of _op instead of op, because op is already be
    // transformed and hard to analysis
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntLower(_op->rhs_) >= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeNeg(makeRoundTowards0Div(op->lhs_, makeNeg(op->rhs_)));
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntLower(_op->lhs_) >= 0) {
        return makeNeg(makeRoundTowards0Div(makeNeg(op->lhs_), op->rhs_));
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    return op;
}

Expr UseBuiltinDiv::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    // Get bounds of children of _op instead of op, because op is already be
    // transformed and hard to analysis
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntLower(_op->rhs_) >= 0) {
        // In case of unsigned data types
        return makeRoundTowards0Div(
            makeAdd(op->lhs_, makeSub(op->rhs_, makeIntConst(1))), op->rhs_);
    }
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntLower(_op->lhs_) >= 0) {
        return makeRoundTowards0Div(op->lhs_, op->rhs_);
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeNeg(makeRoundTowards0Div(op->lhs_, makeNeg(op->rhs_)));
    }
    return op;
}

Expr UseBuiltinDiv::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    // Get bounds of children of _op instead of op, because op is already be
    // transformed and hard to analysis
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntLower(_op->rhs_) >= 0) {
        return makeRemainder(op->lhs_, op->rhs_);
    }
    if (bound_->getIntLower(_op->lhs_) >= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeAdd(
            makeRemainder(op->lhs_, makeSub(makeIntConst(0), op->rhs_)),
            op->rhs_);
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntLower(_op->lhs_) >= 0) {
        return makeAdd(makeRemainder(op->lhs_, op->rhs_), op->rhs_);
    }
    if (bound_->getIntUpper(_op->lhs_) <= 0 &&
        bound_->getIntUpper(_op->rhs_) <= 0) {
        return makeRemainder(op->lhs_, op->rhs_);
    }
    return op;
}

Stmt useBuiltinDiv(const Stmt &_op) { return UseBuiltinDiv()(_op); }

} // namespace freetensor
