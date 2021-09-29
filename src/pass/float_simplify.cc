#include <cmath>

#include <pass/float_simplify.h>

namespace ir {

inline static Expr reduceMul(const std::vector<Expr> &list) {
    Expr ret;
    for (auto &&item : list) {
        ret = ret.isValid() ? makeMul(ret, item) : item;
    }
    return ret;
}

uint64_t FloatSimplify::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

DataType FloatSimplify::dtype(const Expr &op) {
    typeInfer_(op);
    return typeInfer_.types().at(op);
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

    auto trySquare = [this, &squareCnt](std::vector<Expr> &list) {
        for (size_t i = 0; i + 1 < list.size(); i++) {
            for (size_t j = i + 1; j < list.size(); j++) {
                if (this->getHash(list[i]) == this->getHash(list[j])) {
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
    } else {
        isFixPoint_ = false;
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

Stmt FloatSimplify::visit(const VarDef &op) {
    if (buffers_.count(op->name_)) {
        throw InvalidProgram("Nested VarDef with the same name is not allowed");
    }
    buffers_[op->name_] = op->buffer_;
    auto ret = Mutator::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

Expr FloatSimplify::visit(const IntConst &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    constants_[op] = op->val_;
    return op;
}

Expr FloatSimplify::visit(const FloatConst &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloatConst);
    auto op = __op.as<FloatConstNode>();
    constants_[op] = op->val_;
    return op;
}

Expr FloatSimplify::visit(const Add &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(constants_.at(op->lhs_) +
                              constants_.at(op->rhs_));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        isFixPoint_ = false;
        return op->rhs_;
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        isFixPoint_ = false;
        return op->lhs_;
    }

    return op;
}

Expr FloatSimplify::visit(const Sub &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(constants_.at(op->lhs_) -
                              constants_.at(op->rhs_));
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        isFixPoint_ = false;
        return op->lhs_;
    }

    return op;
}

Expr FloatSimplify::visit(const Mul &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(constants_.at(op->lhs_) *
                              constants_.at(op->rhs_));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 1) {
        isFixPoint_ = false;
        return op->rhs_;
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 1) {
        isFixPoint_ = false;
        return op->lhs_;
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }

    return normalizeRealMulDiv(op);
}

Expr FloatSimplify::visit(const RealDiv &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::RealDiv);
    auto op = __op.as<RealDivNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(constants_.at(op->lhs_) /
                              constants_.at(op->rhs_));
    }

    return normalizeRealMulDiv(op);
}

Expr FloatSimplify::visit(const Min &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(
            std::min(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }

    if (op->lhs_->nodeType() == ASTNodeType::Sqrt &&
        op->rhs_->nodeType() == ASTNodeType::Sqrt) {
        isFixPoint_ = false;
        return makeSqrt(makeMin(op->lhs_.as<SqrtNode>()->expr_,
                                op->rhs_.as<SqrtNode>()->expr_));
    }

    return op;
}

Expr FloatSimplify::visit(const Max &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        isFixPoint_ = false;
        return makeFloatConst(
            std::max(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }

    if (op->lhs_->nodeType() == ASTNodeType::Sqrt &&
        op->rhs_->nodeType() == ASTNodeType::Sqrt) {
        isFixPoint_ = false;
        return makeSqrt(makeMax(op->lhs_.as<SqrtNode>()->expr_,
                                op->rhs_.as<SqrtNode>()->expr_));
    }

    return op;
}

Expr FloatSimplify::visit(const Sqrt &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sqrt);
    auto op = __op.as<SqrtNode>();

    if (constants_.count(op->expr_)) {
        isFixPoint_ = false;
        return makeFloatConst(sqrt(constants_.at(op->expr_)));
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
    } else {
        isFixPoint_ = false;
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
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Exp);
    auto op = __op.as<ExpNode>();

    if (constants_.count(op->expr_)) {
        isFixPoint_ = false;
        return makeFloatConst(exp(constants_.at(op->expr_)));
    }

    return op;
}

Expr FloatSimplify::visit(const Square &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Square);
    auto op = __op.as<SquareNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->expr_)) {
        isFixPoint_ = false;
        return makeFloatConst(constants_.at(op->expr_) *
                              constants_.at(op->expr_));
    }

    return op;
}

Expr FloatSimplify::visit(const Abs &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Abs);
    auto op = __op.as<AbsNode>();

    if (!isFloat(dtype(op))) {
        return op;
    }

    if (constants_.count(op->expr_)) {
        isFixPoint_ = false;
        return makeFloatConst(std::abs(constants_.at(op->expr_)));
    }

    return op;
}

Stmt floatSimplify(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        FloatSimplify mutator;
        op = mutator(op);
        if (mutator.isFixPoint() || i > 100) {
            if (i > 100) {
                WARNING(
                    "FloatSimplify iterates over 100 rounds. Maybe there is "
                    "a bug");
            }
            return op;
        }
    }
}

} // namespace ir

