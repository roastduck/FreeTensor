#include <cmath>

#include <hash.h>
#include <pass/float_simplify.h>

namespace freetensor {

inline static Expr reduceMul(const std::vector<Expr> &list) {
    Expr ret;
    for (auto &&item : list) {
        ret = ret.isValid() ? makeMul(ret, item) : item;
    }
    return ret;
}

inline static bool hasSqrt(const Expr &op) {
    if (op->nodeType() == ASTNodeType::Mul) {
        return hasSqrt(op.as<MulNode>()->lhs_) ||
               hasSqrt(op.as<MulNode>()->rhs_);
    } else if (op->nodeType() == ASTNodeType::RealDiv) {
        return hasSqrt(op.as<RealDivNode>()->lhs_) ||
               hasSqrt(op.as<RealDivNode>()->rhs_);
    } else {
        return op->nodeType() == ASTNodeType::Sqrt;
    }
}

bool FloatSimplify::nonNeg(const Expr &op) const {
    if (op->nodeType() == ASTNodeType::IntConst &&
        op.as<IntConstNode>()->val_ >= 0) {
        return true;
    }
    if (op->nodeType() == ASTNodeType::FloatConst &&
        op.as<FloatConstNode>()->val_ >= 0) {
        return true;
    }
    return nonNeg_.count(op);
}

bool FloatSimplify::nonPosi(const Expr &op) const {
    if (op->nodeType() == ASTNodeType::IntConst &&
        op.as<IntConstNode>()->val_ <= 0) {
        return true;
    }
    if (op->nodeType() == ASTNodeType::FloatConst &&
        op.as<FloatConstNode>()->val_ <= 0) {
        return true;
    }
    return nonPosi_.count(op);
}

Expr FloatSimplify::normalizeRealMulDiv(const Expr &op) {
    int sqrtCnt = 0, divCnt = 0, squareCnt = 0;
    std::function<void(const Expr &, std::vector<Expr> &, std::vector<Expr> &,
                       std::vector<Expr> &, std::vector<Expr> &)>
        recur = [&recur, &sqrtCnt,
                 &divCnt](const Expr &expr, std::vector<Expr> &num,
                          std::vector<Expr> &den, std::vector<Expr> &sqrtNum,
                          std::vector<Expr> &sqrtDen) {
            if (expr->nodeType() == ASTNodeType::Mul) {
                recur(expr.as<MulNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<MulNode>()->rhs_, num, den, sqrtNum, sqrtDen);
            } else if (expr->nodeType() == ASTNodeType::RealDiv) {
                recur(expr.as<RealDivNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<RealDivNode>()->rhs_, den, num, sqrtDen, sqrtNum);
                divCnt++;
            } else if (expr->nodeType() == ASTNodeType::Sqrt) {
                sqrtNum.emplace_back(expr.as<SqrtNode>()->expr_);
                sqrtCnt++;
            } else {
                num.emplace_back(expr);
            }
        };
    std::vector<Expr> num, den, sqrtNum, sqrtDen;
    recur(op, num, den, sqrtNum, sqrtDen);

    auto trySquare = [&squareCnt](std::vector<Expr> &list) {
        for (size_t i = 0; i + 1 < list.size(); i++) {
            for (size_t j = i + 1; j < list.size(); j++) {
                if (HashComparator()(list[i], list[j])) {
                    list[i] = makeSquare(list[i]);
                    std::swap(list[j], list.back());
                    list.resize(list.size() - 1);
                    squareCnt++;
                }
            }
        }
    };
    trySquare(num);
    trySquare(den);
    trySquare(sqrtNum);
    trySquare(sqrtDen);
    if (sqrtCnt <= 1 && divCnt <= 1 && squareCnt == 0) {
        return op;
    }

    if (auto x = reduceMul(sqrtNum); x.isValid()) {
        if (auto y = reduceMul(sqrtDen); y.isValid()) {
            num.emplace_back(makeSqrt(makeRealDiv(x, y)));
        } else {
            num.emplace_back(makeSqrt(x));
        }
    } else {
        if (auto y = reduceMul(sqrtDen); y.isValid()) {
            den.emplace_back(makeSqrt(y));
        }
    }
    if (auto x = reduceMul(num); x.isValid()) {
        if (auto y = reduceMul(den); y.isValid()) {
            return makeRealDiv(x, y);
        } else {
            return x;
        }
    } else {
        ASSERT(false); // Impossible
    }
}

Expr FloatSimplify::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
        setNonPosi(op);
    }

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (equals(op->lhs_, 0)) {
        return op->rhs_;
    }
    if (equals(op->rhs_, 0)) {
        return op->lhs_;
    }

    return op;
}

Expr FloatSimplify::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    if (nonNeg(op->lhs_) && nonPosi(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonPosi(op->lhs_) && nonNeg(op->rhs_)) {
        setNonPosi(op);
    }

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (equals(op->lhs_, 0)) {
        // Normalize 0 - x to -1 * x because our ruls for sqrt rely on MulNode
        return makeMul(makeIntConst(-1), op->rhs_);
    }
    if (equals(op->rhs_, 0)) {
        return op->lhs_;
    }

    return op;
}

Expr FloatSimplify::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();
    if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonNeg(op->lhs_) && nonPosi(op->rhs_)) {
        setNonPosi(op);
    }
    if (nonPosi(op->lhs_) && nonNeg(op->rhs_)) {
        setNonPosi(op);
    }
    if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
        setNonNeg(op);
    }

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (equals(op->lhs_, 1)) {
        return op->rhs_;
    }
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->lhs_, 0)) {
        return makeIntConst(0);
    }
    if (equals(op->rhs_, 0)) {
        return makeIntConst(0);
    }

    return normalizeRealMulDiv(op);
}

Expr FloatSimplify::visit(const RealDiv &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::RealDiv);
    auto op = __op.as<RealDivNode>();
    if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonNeg(op->lhs_) && nonPosi(op->rhs_)) {
        setNonPosi(op);
    }
    if (nonPosi(op->lhs_) && nonNeg(op->rhs_)) {
        setNonPosi(op);
    }
    if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
        setNonNeg(op);
    }

    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
    }

    return normalizeRealMulDiv(op);
}

Expr FloatSimplify::visit(const Min &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();
    if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonPosi(op->lhs_) || nonPosi(op->rhs_)) {
        setNonPosi(op);
    }

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (hasSqrt(op->lhs_) && hasSqrt(op->rhs_)) {
        if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
            return makeSqrt(
                makeMin(makeSquare(op->lhs_), makeSquare(op->rhs_)));
        }
        if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
            return makeMul(
                makeIntConst(-1),
                makeSqrt(makeMax(makeSquare(op->lhs_), makeSquare(op->rhs_))));
        }
    }

    return op;
}

Expr FloatSimplify::visit(const Max &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();
    if (nonNeg(op->lhs_) || nonNeg(op->rhs_)) {
        setNonNeg(op);
    }
    if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
        setNonPosi(op);
    }

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (hasSqrt(op->lhs_) && hasSqrt(op->rhs_)) {
        if (nonNeg(op->lhs_) && nonNeg(op->rhs_)) {
            return makeSqrt(
                makeMax(makeSquare(op->lhs_), makeSquare(op->rhs_)));
        }
        if (nonPosi(op->lhs_) && nonPosi(op->rhs_)) {
            return makeMul(
                makeIntConst(-1),
                makeSqrt(makeMin(makeSquare(op->lhs_), makeSquare(op->rhs_))));
        }
    }

    return op;
}

Expr FloatSimplify::visit(const Sqrt &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Sqrt);
    auto op = __op.as<SqrtNode>();
    setNonNeg(op);

    std::function<void(const Expr &, std::vector<Expr> &, std::vector<Expr> &,
                       std::vector<Expr> &, std::vector<Expr> &)>
        recur = [&recur](const Expr &expr, std::vector<Expr> &num,
                         std::vector<Expr> &den, std::vector<Expr> &sqrtNum,
                         std::vector<Expr> &sqrtDen) {
            if (expr->nodeType() == ASTNodeType::Mul) {
                recur(expr.as<MulNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<MulNode>()->rhs_, num, den, sqrtNum, sqrtDen);
            } else if (expr->nodeType() == ASTNodeType::RealDiv) {
                recur(expr.as<RealDivNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<RealDivNode>()->rhs_, den, num, sqrtDen, sqrtNum);
            } else if (expr->nodeType() == ASTNodeType::Square) {
                num.emplace_back(expr.as<SquareNode>()->expr_);
            } else {
                sqrtNum.emplace_back(expr);
            }
        };
    std::vector<Expr> num, den, sqrtNum, sqrtDen;
    recur(op->expr_, num, den, sqrtNum, sqrtDen);
    if (num.empty() && den.empty()) {
        return op;
    }

    if (auto x = reduceMul(sqrtNum); x.isValid()) {
        if (auto y = reduceMul(sqrtDen); y.isValid()) {
            num.emplace_back(makeSqrt(makeRealDiv(x, y)));
        } else {
            num.emplace_back(makeSqrt(x));
        }
    } else {
        if (auto y = reduceMul(sqrtDen); y.isValid()) {
            den.emplace_back(makeSqrt(y));
        }
    }
    if (auto x = reduceMul(num); x.isValid()) {
        if (auto y = reduceMul(den); y.isValid()) {
            return makeRealDiv(x, y);
        } else {
            return x;
        }
    } else {
        ASSERT(false); // Impossible
    }
}

Expr FloatSimplify::visit(const Exp &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Exp);
    auto op = __op.as<ExpNode>();
    setNonNeg(op);
    return op;
}

Expr FloatSimplify::visit(const Square &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Square);
    auto op = __op.as<SquareNode>();
    setNonNeg(op);

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (op->expr_->nodeType() == ASTNodeType::Abs) {
        return makeSquare(op->expr_.as<AbsNode>()->expr_);
    }

    std::function<void(const Expr &, std::vector<Expr> &, std::vector<Expr> &,
                       std::vector<Expr> &, std::vector<Expr> &)>
        recur = [&recur](const Expr &expr, std::vector<Expr> &num,
                         std::vector<Expr> &den, std::vector<Expr> &sqrtNum,
                         std::vector<Expr> &sqrtDen) {
            if (expr->nodeType() == ASTNodeType::Mul) {
                recur(expr.as<MulNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<MulNode>()->rhs_, num, den, sqrtNum, sqrtDen);
            } else if (expr->nodeType() == ASTNodeType::RealDiv) {
                recur(expr.as<RealDivNode>()->lhs_, num, den, sqrtNum, sqrtDen);
                recur(expr.as<RealDivNode>()->rhs_, den, num, sqrtDen, sqrtNum);
            } else if (expr->nodeType() == ASTNodeType::Sqrt) {
                sqrtNum.emplace_back(expr.as<SquareNode>()->expr_);
            } else {
                num.emplace_back(expr);
            }
        };
    std::vector<Expr> num, den, sqrtNum, sqrtDen;
    recur(op->expr_, num, den, sqrtNum, sqrtDen);
    if (sqrtNum.empty() && sqrtDen.empty()) {
        return op;
    }

    if (auto x = reduceMul(num); x.isValid()) {
        if (auto y = reduceMul(den); y.isValid()) {
            sqrtNum.emplace_back(makeSquare(makeRealDiv(x, y)));
        } else {
            sqrtNum.emplace_back(makeSquare(x));
        }
    } else {
        if (auto y = reduceMul(den); y.isValid()) {
            sqrtDen.emplace_back(makeSquare(y));
        }
    }
    if (auto x = reduceMul(sqrtNum); x.isValid()) {
        if (auto y = reduceMul(sqrtDen); y.isValid()) {
            return makeRealDiv(x, y);
        } else {
            return x;
        }
    } else {
        ASSERT(false); // Impossible
    }
}

Expr FloatSimplify::visit(const Abs &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Abs);
    auto op = __op.as<AbsNode>();
    setNonNeg(op);

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (nonNeg(op->expr_)) {
        return op->expr_;
    }

    return op;
}

Stmt floatSimplify(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        FloatSimplify mutator;
        auto newOp = mutator(op);
        if (HashComparator()(newOp, op) || i > 100) {
            if (i > 100) {
                WARNING(
                    "FloatSimplify iterates over 100 rounds. Maybe there is "
                    "a bug");
            }
            return op;
        }
        op = newOp;
    }
}

} // namespace freetensor
