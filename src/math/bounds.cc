#include <algorithm>
#include <climits>
#include <functional>
#include <numeric>
#include <type_traits>

#include <analyze/all_uses.h>
#include <math/bounds.h>
#include <math/utils.h>

namespace freetensor {

static LinearExpr<Rational<int64_t>>
commonDenominator(const LinearExpr<Rational<int64_t>> &_lin) {
    auto lin = _lin;
    auto common = lin.bias_.q_;
    for (auto &&[k, a] : lin.coeff_) {
        common = std::lcm(common, k.q_);
    }
    lin.bias_.p_ *= common / lin.bias_.q_;
    lin.bias_.q_ = common;
    for (auto &[k, a] : lin.coeff_) {
        k.p_ *= common / k.q_;
        k.q_ = common;
    }
    return lin;
}

static Expr linToExprDivisible(const LinearExpr<Rational<int64_t>> &lin) {
    Expr ret;
    for (auto &&item : lin.coeff_) {
        auto k = item.k_;
        auto a = deepCopy(item.a_);

        if (k == 0 || k.p_ % k.q_ != 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k.p_ / k.q_ * a.as<IntConstNode>()->val_);
        } else if (k.p_ / k.q_ == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k.p_ / k.q_), a);
        }

        if (x->nodeType() == ASTNodeType::IntConst &&
            x.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else if (!ret.isValid()) {
            ret = x;
        } else if (x->nodeType() == ASTNodeType::IntConst &&
                   ret->nodeType() == ASTNodeType::IntConst) {
            ret = makeIntConst(x.as<IntConstNode>()->val_ +
                               ret.as<IntConstNode>()->val_);
        } else {
            ret = makeAdd(ret, x);
        }
    }

    if (lin.bias_.p_ % lin.bias_.q_ == 0) {
        Expr b = makeIntConst(lin.bias_.p_ / lin.bias_.q_);
        if (b->nodeType() == ASTNodeType::IntConst &&
            b.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else if (!ret.isValid()) {
            ret = b;
        } else if (b->nodeType() == ASTNodeType::IntConst &&
                   ret->nodeType() == ASTNodeType::IntConst) {
            ret = makeIntConst(b.as<IntConstNode>()->val_ +
                               ret.as<IntConstNode>()->val_);
        } else {
            ret = makeAdd(ret, b);
        }
    }

    return ret;
}

static Expr linToExprNumerator(const LinearExpr<Rational<int64_t>> &lin) {
    Expr ret;
    for (auto &&item : lin.coeff_) {
        auto k = item.k_;
        auto a = deepCopy(item.a_);

        if (k == 0 || k.p_ % k.q_ == 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k.p_ * a.as<IntConstNode>()->val_);
        } else if (k.p_ == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k.p_), a);
        }

        if (x->nodeType() == ASTNodeType::IntConst &&
            x.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else if (!ret.isValid()) {
            ret = x;
        } else if (x->nodeType() == ASTNodeType::IntConst &&
                   ret->nodeType() == ASTNodeType::IntConst) {
            ret = makeIntConst(x.as<IntConstNode>()->val_ +
                               ret.as<IntConstNode>()->val_);
        } else {
            ret = makeAdd(ret, x);
        }
    }

    if (lin.bias_.p_ % lin.bias_.q_ != 0) {
        Expr b = makeIntConst(lin.bias_.p_);
        if (b->nodeType() == ASTNodeType::IntConst &&
            b.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else if (!ret.isValid()) {
            ret = b;
        } else if (b->nodeType() == ASTNodeType::IntConst &&
                   ret->nodeType() == ASTNodeType::IntConst) {
            ret = makeIntConst(b.as<IntConstNode>()->val_ +
                               ret.as<IntConstNode>()->val_);
        } else {
            ret = makeAdd(ret, b);
        }
    }

    return ret;
}

const Expr &UpperBound::expr() {
    if (expr_.isValid()) {
        return expr_;
    }
    auto cdLin = commonDenominator(lin_);
    auto divisible = linToExprDivisible(cdLin);
    auto nonDivisible = linToExprNumerator(cdLin);
    if (nonDivisible.isValid()) {
        if (nonDivisible->nodeType() == ASTNodeType::IntConst) {
            nonDivisible = makeIntConst(floorDiv(
                nonDivisible.as<IntConstNode>()->val_, cdLin.bias_.q_));
        } else {
            nonDivisible =
                makeFloorDiv(nonDivisible, makeIntConst(cdLin.bias_.q_));
        }
        expr_ = divisible.isValid() ? makeAdd(divisible, nonDivisible)
                                    : nonDivisible;
    } else {
        expr_ = divisible.isValid() ? divisible : makeIntConst(0);
    }
    return expr_;
}

const Expr &LowerBound::expr() {
    if (expr_.isValid()) {
        return expr_;
    }
    auto cdLin = commonDenominator(lin_);
    auto divisible = linToExprDivisible(cdLin);
    auto nonDivisible = linToExprNumerator(cdLin);
    if (nonDivisible.isValid()) {
        if (nonDivisible->nodeType() == ASTNodeType::IntConst) {
            nonDivisible = makeIntConst(
                ceilDiv(nonDivisible.as<IntConstNode>()->val_, cdLin.bias_.q_));
        } else {
            nonDivisible =
                makeCeilDiv(nonDivisible, makeIntConst(cdLin.bias_.q_));
        }
        expr_ = divisible.isValid() ? makeAdd(divisible, nonDivisible)
                                    : nonDivisible;
    } else {
        expr_ = divisible.isValid() ? divisible : makeIntConst(0);
    }
    return expr_;
}

const std::unordered_set<std::string> &UpperBound::allNames() {
    if (allNames_.has_value()) {
        return *allNames_;
    }
    allNames_ = std::make_optional(lin_.allNames());
    return *allNames_;
}

const std::unordered_set<std::string> &LowerBound::allNames() {
    if (allNames_.has_value()) {
        return *allNames_;
    }
    allNames_ = std::make_optional(lin_.allNames());
    return *allNames_;
}

UpperBound add(const UpperBound &b1, const UpperBound &b2) {
    return add(b1.lin(), b2.lin());
}
LowerBound add(const LowerBound &b1, const LowerBound &b2) {
    return add(b1.lin(), b2.lin());
}

UpperBound sub(const UpperBound &b1, const LowerBound &b2) {
    return sub(b1.lin(), b2.lin());
}
LowerBound sub(const LowerBound &b1, const UpperBound &b2) {
    return sub(b1.lin(), b2.lin());
}

UpperBound mul(const UpperBound &b, int k) {
    return mul(b.lin(), Rational<int64_t>(k));
}
LowerBound mul(const LowerBound &b, int k) {
    return mul(b.lin(), Rational<int64_t>(k));
}

UpperBound floorDiv(const UpperBound &b, int k) {
    auto ret = b.lin();
    for (auto &&item : ret.coeff_) {
        item.k_ /= k;
    }
    ret.bias_ /= k;
    return UpperBound(std::move(ret));
}
LowerBound floorDiv(const LowerBound &b, int k) {
    auto ret = b.lin();
    for (auto &&item : ret.coeff_) {
        item.k_ /= k;
    }
    ret.bias_ -= k - 1;
    ret.bias_ /= k;
    return LowerBound(std::move(ret));
}

UpperBound ceilDiv(const UpperBound &b, int k) {
    auto ret = b.lin();
    for (auto &&item : ret.coeff_) {
        item.k_ /= k;
    }
    ret.bias_ += k - 1;
    ret.bias_ /= k;
    return UpperBound(std::move(ret));
}
LowerBound ceilDiv(const LowerBound &b, int k) {
    auto ret = b.lin();
    for (auto &&item : ret.coeff_) {
        item.k_ /= k;
    }
    ret.bias_ /= k;
    return LowerBound(std::move(ret));
}

bool alwaysLT(const UpperBound &b1, const LowerBound &b2) {
    // Case 1: lower and upper round to some const
    if (b1.lin().isConst() && b2.lin().isConst() &&
        floorDiv(b1.lin().bias_.p_, b1.lin().bias_.q_) <
            ceilDiv(b2.lin().bias_.p_, b2.lin().bias_.q_)) {
        return true;
    }

    // Case 2: upper < lower
    return b1.lin().bias_ < b2.lin().bias_ &&
           hasIdenticalCoeff(b1.lin(), b2.lin());
}
bool alwaysLE(const UpperBound &b1, const LowerBound &b2) {
    // Case 1: lower and upper round to some const
    if (b1.lin().isConst() && b2.lin().isConst() &&
        floorDiv(b1.lin().bias_.p_, b1.lin().bias_.q_) <=
            ceilDiv(b2.lin().bias_.p_, b2.lin().bias_.q_)) {
        return true;
    }

    // Case 2: upper - lower < 1
    // E.g. x <= a + 2/3 and y >= a ==> (int)y <= (int)x
    return b1.lin().bias_ - b2.lin().bias_ < 1 &&
           hasIdenticalCoeff(b1.lin(), b2.lin());
}

} // namespace freetensor
