#include <algorithm>
#include <climits>
#include <functional>

#include <math/bounds.h>

namespace ir {

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
    Expr b = makeIntConst(lin.bias_.p_);
    // Difference between UpperBound and LowerBound
    b = lin.bias_.q_ == 1 ? b : makeFloorDiv(b, makeIntConst(lin.bias_.q_));

    for (auto &&item : lin.coeff_) {
        auto k = item.second.k_;
        auto &&a = item.second.a_;

        if (k == 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k.p_ * a.as<IntConstNode>()->val_);
        } else if (k == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k.p_), a);
        }

        // Difference between UpperBound and LowerBound
        x = k.q_ == 1 ? x : makeFloorDiv(x, makeIntConst(k.q_));

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
    expr_ = std::move(b);
}

LowerBound::LowerBound(const Expr &expr)
    : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}

LowerBound::LowerBound(const LinearExpr<Rational<int>> &lin) : lin_(lin) {
    Expr b = makeIntConst(lin.bias_.p_);
    // Difference between UpperBound and LowerBound
    b = lin.bias_.q_ == 1 ? b : makeCeilDiv(b, makeIntConst(lin.bias_.q_));

    for (auto &&item : lin.coeff_) {
        auto k = item.second.k_;
        auto &&a = item.second.a_;

        if (k == 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k.p_ * a.as<IntConstNode>()->val_);
        } else if (k == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k.p_), a);
        }

        // Difference between UpperBound and LowerBound
        x = k.q_ == 1 ? x : makeCeilDiv(x, makeIntConst(k.q_));

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
    expr_ = std::move(b);
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

