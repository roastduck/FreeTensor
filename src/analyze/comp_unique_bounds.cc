#include <algorithm>
#include <climits>

#include <analyze/all_uses.h>
#include <analyze/analyze_linear.h>
#include <analyze/comp_unique_bounds.h>
#include <container_utils.h>

namespace freetensor {

void CompUniqueBounds::updLower(LowerBoundsList &list,
                                const LowerBound &bound) const {
    for (LowerBound &old : list) {
        // The same .expr() does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin() == bound.lin()) {
            return;
        }
        if (hasIdenticalCoeff(bound.lin(), old.lin())) {
            auto oldVal = old.lin().bias_;
            auto newVal = bound.lin().bias_;
            if (newVal > oldVal) {
                old = LowerBound(
                    LinearExpr<Rational<int64_t>>{old.lin().coeff_, newVal});
            }
            return;
        }
    }
    list.emplace_back(bound);
}

void CompUniqueBounds::updUpper(UpperBoundsList &list,
                                const UpperBound &bound) const {
    for (UpperBound &old : list) {
        // The same .expr() does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin() == bound.lin()) {
            return;
        }
        if (hasIdenticalCoeff(bound.lin(), old.lin())) {
            auto oldVal = old.lin().bias_;
            auto newVal = bound.lin().bias_;
            if (newVal < oldVal) {
                old = UpperBound(
                    LinearExpr<Rational<int64_t>>{old.lin().coeff_, newVal});
            }
            return;
        }
    }
    list.emplace_back(bound);
}

int CompUniqueBounds::getIntLower(const Expr &op) {
    int ret = INT_MIN;
    for (auto &&b : getLower(op)) {
        if (b.lin().isConst()) {
            auto bias = b.lin().bias_;
            ret =
                std::max(ret, (int)ceilDiv(bias.p_, bias.q_)); // FIXME: int64_t
        }
    }
    return ret;
}

int CompUniqueBounds::getIntUpper(const Expr &op) {
    int ret = INT_MAX;
    for (auto &&b : getUpper(op)) {
        if (b.lin().isConst()) {
            auto bias = b.lin().bias_;
            ret = std::min(ret,
                           (int)floorDiv(bias.p_, bias.q_)); // FIXME: int64_t
        }
    }
    return ret;
}

Opt<int> CompUniqueBounds::getInt(const Expr &op) {
    int lower = getIntLower(op);
    int upper = getIntUpper(op);
    return lower == upper ? Opt<int>::make(lower) : nullptr;
}

bool CompUniqueBounds::alwaysLT(const Expr &lhs, const Expr &rhs) {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (freetensor::alwaysLT(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

bool CompUniqueBounds::alwaysLE(const Expr &lhs, const Expr &rhs) {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (freetensor::alwaysLE(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

void CompUniqueBounds::visitExpr(const Expr &op) {
    if (lower_.count(op) || upper_.count(op)) {
        return;
    }
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    lower = {};
    upper = {};

    if (!isInt(dtype(op))) {
        return;
    }

    BaseClass::visitExpr(op);
    auto tr = transients_.transient(op);
    for (auto &&first : tr.lower_) {
        for (auto &&item : getLower(first)) {
            if (!hasIntersect(allNames(op, true),
                              allNames(item.expr(), true))) {
                // No loop bounds: X cannot bound X itself
                updLower(lower, item);
            }
        }
    }
    for (auto &&second : tr.upper_) {
        for (auto &&item : getUpper(second)) {
            if (!hasIntersect(allNames(op, true),
                              allNames(item.expr(), true))) {
                // No loop bounds: X cannot bound X itself
                updUpper(upper, item);
            }
        }
    }
}

void CompUniqueBounds::visit(const Var &op) {
    BaseClass::visit(op);
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
}

void CompUniqueBounds::visit(const Load &op) {
    BaseClass::visit(op);
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
}

void CompUniqueBounds::visit(const IntConst &op) {
    BaseClass::visit(op);
    updLower(lower_[op],
             LowerBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
    updUpper(upper_[op],
             UpperBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
}

void CompUniqueBounds::visitLinear(const Expr &op) {
    auto &lower = lower_[op];
    auto &upper = upper_[op];

    auto &&lin = linear(op);

    // lin is op itself. Stop self-recursion
    if (lin.bias_ == 0 && lin.coeff_.size() == 1 &&
        HashComparator()(lin.coeff_.front().a_, op)) {
        return;
    }

    LowerBoundsList retLower = {
        LowerBound{LinearExpr<Rational<int64_t>>{{}, lin.bias_}}};
    UpperBoundsList retUpper = {
        UpperBound{LinearExpr<Rational<int64_t>>{{}, lin.bias_}}};
    for (auto &&[k, a] : lin.coeff_) {
        LowerBoundsList itemLower = {};
        UpperBoundsList itemUpper = {};
        if (k > 0) {
            for (auto &&b : getLower(a)) {
                updLower(itemLower, mul(b, k));
            }
            for (auto &&b : getUpper(a)) {
                updUpper(itemUpper, mul(b, k));
            }
            if (a->nodeType() == ASTNodeType::FloorDiv) {
                auto div = a.as<FloorDivNode>();
                if (auto k1 = getInt(div->rhs_);
                    k1.isValid() && *k1 > 0 && k % *k1 == 0) {
                    auto equ =
                        makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_));
                    for (auto &&b : getLower(equ)) {
                        updLower(itemLower, mul(b, k / *k1));
                    }
                    for (auto &&b : getUpper(equ)) {
                        updUpper(itemUpper, mul(b, k / *k1));
                    }
                }
            }
        } else {
            for (auto &&b : getLower(a)) {
                updUpper(itemUpper, mul(UpperBound{b.lin()}, k));
            }
            for (auto &&b : getUpper(a)) {
                updLower(itemLower, mul(LowerBound{b.lin()}, k));
            }
            if (a->nodeType() == ASTNodeType::FloorDiv) {
                auto div = a.as<FloorDivNode>();
                if (auto k1 = getInt(div->rhs_);
                    k1.isValid() && *k1 > 0 && k % *k1 == 0) {
                    auto equ =
                        makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_));
                    for (auto &&b : getLower(equ)) {
                        updUpper(itemUpper, mul(UpperBound{b.lin()}, k / *k1));
                    }
                    for (auto &&b : getUpper(equ)) {
                        updLower(itemLower, mul(LowerBound{b.lin()}, k / *k1));
                    }
                }
            }
        }

        LowerBoundsList newLower = {};
        UpperBoundsList newUpper = {};
        for (auto &&b1 : retLower) {
            for (auto &&b2 : itemLower) {
                updLower(newLower, add(b1, b2));
            }
        }
        for (auto &&b1 : retUpper) {
            for (auto &&b2 : itemUpper) {
                updUpper(newUpper, add(b1, b2));
            }
        }
        retLower = std::move(newLower);
        retUpper = std::move(newUpper);
    }
    lower = std::move(retLower);
    upper = std::move(retUpper);
}

void CompUniqueBounds::visit(const Add &op) {
    // no need to recurse. getLower or getUpper recurses
    visitLinear(op);
}

void CompUniqueBounds::visit(const Sub &op) {
    // no need to recurse. getLower or getUpper recurses
    visitLinear(op);
}

void CompUniqueBounds::visit(const Mul &op) {
    // no need to recurse. getLower or getUpper recurses
    visitLinear(op);
}

void CompUniqueBounds::visit(const Square &op) {
    // no need to recurse. getLower or getUpper recurses

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->expr_); k.isValid()) {
        updLower(lower, LowerBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
        updUpper(upper, UpperBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
    }
}

void CompUniqueBounds::visit(const FloorDiv &op) {
    // no need to recurse. getLower or getUpper recurses

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(lower, floorDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper, floorDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper, floorDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower, floorDiv(LowerBound{b.lin()}, *k));
            }
        }
    }
}

void CompUniqueBounds::visit(const CeilDiv &op) {
    // no need to recurse. getLower or getUpper recurses

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(lower, ceilDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper, ceilDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper, ceilDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower, ceilDiv(LowerBound{b.lin()}, *k));
            }
        }
    }
}

void CompUniqueBounds::visit(const Mod &op) {
    // no need to recurse. getLower or getUpper recurses
    if (auto &&l = getInt(op->lhs_); l.isValid()) {
        if (auto &&r = getInt(op->rhs_); r.isValid()) {
            updLower(lower_[op], LowerBound{LinearExpr<Rational<int64_t>>{
                                     {}, mod(*l, *r)}});
            updUpper(upper_[op], UpperBound{LinearExpr<Rational<int64_t>>{
                                     {}, mod(*l, *r)}});
            return;
        }
    }
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
    updLower(lower_[op], LowerBound{LinearExpr<Rational<int64_t>>{{}, 0}});
    for (auto &&item : getUpper(op->rhs_)) {
        updUpper(upper_[op], item);
    }
}

void CompUniqueBounds::visit(const Min &op) {
    // no need to recurse. getLower or getUpper recurses
    auto &lower = lower_[op];
    auto &upper = upper_[op];

    std::vector<Expr> oper, all;
    std::function<void(const Expr &)> recur = [&](const Expr &expr) {
        if (expr->nodeType() == ASTNodeType::Min) {
            recur(expr.as<MinNode>()->lhs_);
            recur(expr.as<MinNode>()->rhs_);
        } else {
            oper.emplace_back(expr);
        }
    };
    recur(op);
    for (auto &&next : oper) {
        for (auto &old : all) {
            auto diff = makeSub(old, next);
            if (getIntUpper(diff) <= 0) {
                goto ignore;
            }
            if (getIntLower(diff) >= 0) {
                old = next;
                goto ignore;
            }
        }
        all.emplace_back(next);
    ignore:;
    }
    if (all.size() == 1) {
        lower = getLower(all.front());
        upper = getUpper(all.front());
        return;
    }

    bool hasConstLower = true;
    Opt<Rational<int64_t>> constLower;
    for (auto &&item : all) {
        for (auto &&b : getUpper(item)) {
            updUpper(upper, b);
        }
        if (hasConstLower) {
            for (auto &&b : getLower(item)) {
                if (b.lin().isConst()) {
                    if (constLower.isValid()) {
                        *constLower = std::min(*constLower, b.lin().bias_);
                    } else {
                        constLower =
                            Opt<Rational<int64_t>>::make(b.lin().bias_);
                    }
                    goto done;
                }
            }
            hasConstLower = false;
        done:;
        }
    }
    if (hasConstLower && constLower.isValid()) {
        updLower(lower, LinearExpr<Rational<int64_t>>{{}, *constLower});
    }

    if (all.size() == oper.size()) {
        updLower(lower, LowerBound{op});
        updUpper(upper, UpperBound{op});
    } else {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMin(ret, item) : item;
        }
        updLower(lower, LowerBound{ret});
        updUpper(upper, UpperBound{ret});
    }
}

void CompUniqueBounds::visit(const Max &op) {
    // no need to recurse. getLower or getUpper recurses
    auto &lower = lower_[op];
    auto &upper = upper_[op];

    std::vector<Expr> oper, all;
    std::function<void(const Expr &)> recur = [&](const Expr &expr) {
        if (expr->nodeType() == ASTNodeType::Max) {
            recur(expr.as<MaxNode>()->lhs_);
            recur(expr.as<MaxNode>()->rhs_);
        } else {
            oper.emplace_back(expr);
        }
    };
    recur(op);
    for (auto &&next : oper) {
        for (auto &old : all) {
            auto diff = makeSub(old, next);
            if (getIntLower(diff) >= 0) {
                goto ignore;
            }
            if (getIntUpper(diff) <= 0) {
                old = next;
                goto ignore;
            }
        }
        all.emplace_back(next);
    ignore:;
    }
    if (all.size() == 1) {
        lower = getLower(all.front());
        upper = getUpper(all.front());
        return;
    }

    bool hasConstUpper = true;
    Opt<Rational<int64_t>> constUpper;
    for (auto &&item : all) {
        for (auto &&b : getLower(item)) {
            updLower(lower, b);
        }
        if (hasConstUpper) {
            for (auto &&b : getUpper(item)) {
                if (b.lin().isConst()) {
                    if (constUpper.isValid()) {
                        *constUpper = std::max(*constUpper, b.lin().bias_);
                    } else {
                        constUpper =
                            Opt<Rational<int64_t>>::make(b.lin().bias_);
                    }
                    goto done;
                }
            }
            hasConstUpper = false;
        done:;
        }
    }
    if (hasConstUpper && constUpper.isValid()) {
        updUpper(upper, LinearExpr<Rational<int64_t>>{{}, *constUpper});
    }

    if (all.size() == oper.size()) {
        updLower(lower, LowerBound{op});
        updUpper(upper, UpperBound{op});
    } else {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMax(ret, item) : item;
        }
        updLower(lower, LowerBound{ret});
        updUpper(upper, UpperBound{ret});
    }
}

void CompUniqueBounds::visit(const IfExpr &op) {
    // no need to recurse. getLower or getUpper recurses
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b1 : getUpper(op->thenCase_)) {
        for (auto &&b2 : getUpper(op->elseCase_)) {
            if (b1.lin().isConst() && b2.lin().isConst()) {
                updUpper(upper,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::max(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    for (auto &&b1 : getLower(op->thenCase_)) {
        for (auto &&b2 : getLower(op->elseCase_)) {
            if (b1.lin().isConst() && b2.lin().isConst()) {
                updLower(lower,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::min(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower, LowerBound{op});
    updUpper(upper, UpperBound{op});
}

} // namespace freetensor
