#include <algorithm>
#include <climits>
#include <functional>

#include <math/bounds.h>
#include <math/utils.h>

namespace ir {

static LinearExpr<Rational<int>>
commonDenominator(const LinearExpr<Rational<int>> &_lin) {
    auto lin = _lin;
    auto common = lin.bias_.q_;
    for (auto &&item : lin.coeff_) {
        common = lcm(common, item.second.k_.q_);
    }
    lin.bias_.p_ *= common / lin.bias_.q_;
    lin.bias_.q_ = common;
    for (auto &item : lin.coeff_) {
        item.second.k_.p_ *= common / item.second.k_.q_;
        item.second.k_.q_ = common;
    }
    return lin;
}

static Expr linToExprNumerator(const LinearExpr<Rational<int>> &lin) {
    Expr b = makeIntConst(lin.bias_.p_);

    for (auto &&item : lin.coeff_) {
        auto k = item.second.k_;
        auto &&a = item.second.a_;

        if (k == 0) {
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
            b->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(x.as<IntConstNode>()->val_ +
                            b.as<IntConstNode>()->val_);
        } else if (b->nodeType() == ASTNodeType::IntConst &&
                    b.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else {
            x = makeAdd(x, b);
        }

        b = std::move(x);
    }

    return b;
}

template <class T> static T addImpl(const T &b1, const T &b2) {
    auto ret = b1.lin_;
    for (auto &&item : b2.lin_.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k_ += item.second.k_;
        } else {
            ret.coeff_[item.first] = item.second;
        }
    }
    ret.bias_ += b2.lin_.bias_;
    return ret;
}

template <class T, class U> static T subImpl(const T &b1, const U &b2) {
    auto ret = b1.lin_;
    for (auto &&item : b2.lin_.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k_ -= item.second.k_;
        } else {
            ret.coeff_[item.first] = {-item.second.k_, item.second.a_};
        }
    }
    ret.bias_ -= b2.lin_.bias_;
    return ret;
}

template <class T> static T mulImpl(const T &b, int k) {
    auto ret = b.lin_;
    for (auto &&item : ret.coeff_) {
        item.second.k_ *= k;
    }
    ret.bias_ *= k;
    return ret;
}

UpperBound::UpperBound(const Expr &expr)
    : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}

UpperBound::UpperBound(const LinearExpr<Rational<int>> &lin) : lin_(lin) {
    auto cdLin = commonDenominator(lin);
    expr_ = linToExprNumerator(cdLin);
    if (cdLin.bias_.q_ != 1) {
        if (expr_->nodeType() == ASTNodeType::IntConst) {
            expr_ = makeIntConst(
                floorDiv(expr_.as<IntConstNode>()->val_, cdLin.bias_.q_));
        } else {
            expr_ = makeFloorDiv(expr_, makeIntConst(cdLin.bias_.q_));
        }
    }
}

LowerBound::LowerBound(const Expr &expr)
    : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}

LowerBound::LowerBound(const LinearExpr<Rational<int>> &lin) : lin_(lin) {
    auto cdLin = commonDenominator(lin);
    expr_ = linToExprNumerator(cdLin);
    if (cdLin.bias_.q_ != 1) {
        if (expr_->nodeType() == ASTNodeType::IntConst) {
            expr_ = makeIntConst(
                ceilDiv(expr_.as<IntConstNode>()->val_, cdLin.bias_.q_));
        } else {
            expr_ = makeCeilDiv(expr_, makeIntConst(cdLin.bias_.q_));
        }
    }
}

UpperBound add(const UpperBound &b1, const UpperBound &b2) {
    return addImpl(b1, b2);
}
LowerBound add(const LowerBound &b1, const LowerBound &b2) {
    return addImpl(b1, b2);
}

UpperBound sub(const UpperBound &b1, const LowerBound &b2) {
    return subImpl(b1, b2);
}
LowerBound sub(const LowerBound &b1, const UpperBound &b2) {
    return subImpl(b1, b2);
}

UpperBound mul(const UpperBound &b, int k) { return mulImpl(b, k); }
LowerBound mul(const LowerBound &b, int k) { return mulImpl(b, k); }

UpperBound floorDiv(const UpperBound &b, int k) {
    auto ret = b.lin_;
    for (auto &&item : ret.coeff_) {
        item.second.k_ /= k;
    }
    ret.bias_ /= k;
    return ret;
}
LowerBound floorDiv(const LowerBound &b, int k) {
    auto ret = b.lin_;
    for (auto &&item : ret.coeff_) {
        item.second.k_ /= k;
    }
    ret.bias_ -= k - 1;
    ret.bias_ /= k;
    return ret;
}

UpperBound ceilDiv(const UpperBound &b, int k) {
    auto ret = b.lin_;
    for (auto &&item : ret.coeff_) {
        item.second.k_ /= k;
    }
    ret.bias_ += k - 1;
    ret.bias_ /= k;
    return ret;
}
LowerBound ceilDiv(const LowerBound &b, int k) {
    auto ret = b.lin_;
    for (auto &&item : ret.coeff_) {
        item.second.k_ /= k;
    }
    ret.bias_ /= k;
    return ret;
}

} // namespace ir

