#include <cmath>

#include <hash.h>
#include <math/utils.h>
#include <pass/float_simplify.h>
#include <pass/refine_sign_data_type.h>

namespace freetensor {

inline static std::pair<double, Expr> reduceMul(const std::vector<Expr> &list) {
    double c = 1;
    Expr ret;
    for (auto &&item : list) {
        if (item->nodeType() == ASTNodeType::IntConst) {
            c *= item.as<IntConstNode>()->val_;
        } else if (item->nodeType() == ASTNodeType::FloatConst) {
            c *= item.as<FloatConstNode>()->val_;
        } else {
            ret = ret.isValid() ? makeMul(ret, item) : item;
        }
    }
    return {c, ret};
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

Expr FloatSimplify::normalizeRealMulDiv(const Expr &op) {
    int sqrtCnt = 0, divCnt = 0, squareCnt = 0, constCnt = 0;
    std::function<void(const Expr &, std::vector<Expr> &, std::vector<Expr> &,
                       std::vector<Expr> &, std::vector<Expr> &)>
        recur = [&recur, &sqrtCnt, &divCnt,
                 &constCnt](const Expr &expr, std::vector<Expr> &num,
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
                if (expr->isConst()) {
                    constCnt++;
                }
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
    if (sqrtCnt <= 1 && divCnt <= 1 && squareCnt == 0 && constCnt <= 1) {
        return op;
    }

    {
        auto &&[cx, x] = reduceMul(sqrtNum);
        auto &&[cy, y] = reduceMul(sqrtDen);
        if (cx / cy != 1) {
            num.emplace_back(makeFloatConst(sqrt(cx / cy)));
        }
        if (x.isValid()) {
            if (y.isValid()) {
                num.emplace_back(makeSqrt(makeRealDiv(x, y)));
            } else {
                num.emplace_back(makeSqrt(x));
            }
        } else {
            if (y.isValid()) {
                den.emplace_back(makeSqrt(y));
            }
        }
    }
    {
        auto &&[cx, x] = reduceMul(num);
        auto &&[cy, y] = reduceMul(den);
        if (x.isValid()) {
            if (y.isValid()) {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), makeRealDiv(x, y));
                } else {
                    return makeRealDiv(x, y);
                }
            } else {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), x);
                } else {
                    return x;
                }
            }
        } else {
            if (y.isValid()) {
                return makeRealDiv(makeFloatConst(cx / cy), y);
            } else {
                return makeFloatConst(cx / cy);
            }
        }
    }
}

Expr FloatSimplify::visitExpr(const Expr &_expr) {
    auto expr = BaseClass::visitExpr(_expr);

    if (expr->isBinary()) {
        auto &&lhs = expr.as<BinaryExprNode>()->lhs_;
        auto &&rhs = expr.as<BinaryExprNode>()->rhs_;

        // Fold operands into `IfExpr` to enble further foldings or other
        // simplifications. This only happens if either or both sides are
        // constants, to avoid combination explosion
        if (lhs->nodeType() == ASTNodeType::IfExpr) {
            auto &&branch = lhs.as<IfExprNode>();
            if ((branch->thenCase_->isConst() &&
                 branch->elseCase_->isConst()) ||
                rhs->isConst()) {
                return makeIfExpr(
                    branch->cond_,
                    makeBinary(expr->nodeType(), branch->thenCase_, rhs),
                    makeBinary(expr->nodeType(), branch->elseCase_, rhs));
            }
        }
        if (rhs->nodeType() == ASTNodeType::IfExpr) {
            auto &&branch = rhs.as<IfExprNode>();
            if ((branch->thenCase_->isConst() &&
                 branch->elseCase_->isConst()) ||
                lhs->isConst()) {
                return makeIfExpr(
                    branch->cond_,
                    makeBinary(expr->nodeType(), lhs, branch->thenCase_),
                    makeBinary(expr->nodeType(), lhs, branch->elseCase_));
            }
        }
    }

    return expr;
}

Expr FloatSimplify::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();

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

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (hasSqrt(op->lhs_) && hasSqrt(op->rhs_)) {
        if (isGE0(op->lhs_->dtype()) && isGE0(op->rhs_->dtype())) {
            return makeSqrt(
                makeMin(makeSquare(op->lhs_), makeSquare(op->rhs_)));
        }
        if (isLE0(op->lhs_->dtype()) && isLE0(op->rhs_->dtype())) {
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

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (hasSqrt(op->lhs_) && hasSqrt(op->rhs_)) {
        if (isGE0(op->lhs_->dtype()) && isGE0(op->rhs_->dtype())) {
            return makeSqrt(
                makeMax(makeSquare(op->lhs_), makeSquare(op->rhs_)));
        }
        if (isLE0(op->lhs_->dtype()) && isLE0(op->rhs_->dtype())) {
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

    {
        auto &&[cx, x] = reduceMul(sqrtNum);
        auto &&[cy, y] = reduceMul(sqrtDen);
        if (cx / cy != 1) {
            num.emplace_back(makeFloatConst(sqrt(cx / cy)));
        }
        if (x.isValid()) {
            if (y.isValid()) {
                num.emplace_back(makeSqrt(makeRealDiv(x, y)));
            } else {
                num.emplace_back(makeSqrt(x));
            }
        } else {
            if (y.isValid()) {
                den.emplace_back(makeSqrt(y));
            }
        }
    }
    {
        auto &&[cx, x] = reduceMul(num);
        auto &&[cy, y] = reduceMul(den);
        if (x.isValid()) {
            if (y.isValid()) {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), makeRealDiv(x, y));
                } else {
                    return makeRealDiv(x, y);
                }
            } else {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), x);
                } else {
                    return x;
                }
            }
        } else {
            if (y.isValid()) {
                return makeRealDiv(makeFloatConst(cx / cy), y);
            } else {
                return makeFloatConst(cx / cy);
            }
        }
    }
}

Expr FloatSimplify::visit(const Square &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Square);
    auto op = __op.as<SquareNode>();

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

    {
        auto &&[cx, x] = reduceMul(num);
        auto &&[cy, y] = reduceMul(den);
        if (cx / cy != 1) {
            sqrtNum.emplace_back(makeFloatConst(square(cx / cy)));
        }
        if (x.isValid()) {
            if (y.isValid()) {
                sqrtNum.emplace_back(makeSquare(makeRealDiv(x, y)));
            } else {
                sqrtNum.emplace_back(makeSquare(x));
            }
        } else {
            if (y.isValid()) {
                sqrtDen.emplace_back(makeSquare(y));
            }
        }
    }
    {
        auto &&[cx, x] = reduceMul(sqrtNum);
        auto &&[cy, y] = reduceMul(sqrtDen);
        if (x.isValid()) {
            if (y.isValid()) {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), makeRealDiv(x, y));
                } else {
                    return makeRealDiv(x, y);
                }
            } else {
                if (cx / cy != 1) {
                    return makeMul(makeFloatConst(cx / cy), x);
                } else {
                    return x;
                }
            }
        } else {
            if (y.isValid()) {
                return makeRealDiv(makeFloatConst(cx / cy), y);
            } else {
                return makeFloatConst(cx / cy);
            }
        }
    }
}

Expr FloatSimplify::visit(const Abs &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Abs);
    auto op = __op.as<AbsNode>();

    if (!isFloat(op->dtype())) {
        return op;
    }

    if (isGE0(op->expr_->dtype())) {
        return op->expr_;
    }

    return op;
}

Stmt floatSimplify(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = refineSignDataType(op); // Do this every iterations
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
