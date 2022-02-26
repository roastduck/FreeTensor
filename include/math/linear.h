#ifndef LINEAR_H
#define LINEAR_H

#include <algorithm>
#include <iostream>

#include <hash.h>

namespace ir {

/**
 * k * a
 */
template <class T> struct Scale {
    T k_;
    Expr a_;
};

/**
 * (sum_i k_i * a_i) + b
 */
template <class T> struct LinearExpr {
    // std::unordered_map can not guarantee ASTs generated from two identical
    // `LinearExpr`s are the same, but std::map is too slow. So, we are using
    // std::vector and sort each factor by its hash
    std::vector<Scale<T>> coeff_;
    T bias_;
};

template <class T>
LinearExpr<T> add(const LinearExpr<T> &lhs, const LinearExpr<T> &rhs) {
    LinearExpr<T> ret;
    auto m = lhs.coeff_.size(), n = rhs.coeff_.size();
    ret.coeff_.reserve(m + n);
    for (size_t p = 0, q = 0; p < m || q < n;) {
        if (q == n ||
            (p < m && lhs.coeff_[p].a_->hash() < rhs.coeff_[q].a_->hash())) {
            ret.coeff_.emplace_back(lhs.coeff_[p++]);
        } else if (p == m || (q < n && lhs.coeff_[p].a_->hash() >
                                           rhs.coeff_[q].a_->hash())) {
            ret.coeff_.emplace_back(rhs.coeff_[q++]);
        } else {
            Scale<T> s{lhs.coeff_[p].k_ + rhs.coeff_[q].k_, lhs.coeff_[p].a_};
            p++, q++;
            if (s.k_ != 0) {
                ret.coeff_.emplace_back(s);
            }
        }
    }
    ret.bias_ = lhs.bias_ + rhs.bias_;
    return ret;
}

template <class T>
LinearExpr<T> sub(const LinearExpr<T> &lhs, const LinearExpr<T> &rhs) {
    LinearExpr<T> ret;
    auto m = lhs.coeff_.size(), n = rhs.coeff_.size();
    ret.coeff_.reserve(m + n);
    for (size_t p = 0, q = 0; p < m || q < n;) {
        if (q == n ||
            (p < m && lhs.coeff_[p].a_->hash() < rhs.coeff_[q].a_->hash())) {
            ret.coeff_.emplace_back(lhs.coeff_[p++]);
        } else if (p == m || (q < n && lhs.coeff_[p].a_->hash() >
                                           rhs.coeff_[q].a_->hash())) {
            ret.coeff_.emplace_back(rhs.coeff_[q++]);
            ret.coeff_.back().k_ = -ret.coeff_.back().k_;
        } else {
            Scale<T> s{lhs.coeff_[p].k_ - rhs.coeff_[q].k_, lhs.coeff_[p].a_};
            p++, q++;
            if (s.k_ != 0) {
                ret.coeff_.emplace_back(s);
            }
        }
    }
    ret.bias_ = lhs.bias_ - rhs.bias_;
    return ret;
}

template <class T> LinearExpr<T> mul(const LinearExpr<T> &lin, const T &k) {
    if (k == 0) {
        return LinearExpr<T>{{}, 0};
    }
    LinearExpr<T> ret;
    ret.coeff_.reserve(lin.coeff_.size());
    for (auto &&item : lin.coeff_) {
        ret.coeff_.emplace_back(Scale<T>{item.k_ * k, item.a_});
    }
    ret.bias_ = lin.bias_ * k;
    return ret;
}

template <class T>
bool hasIdenticalCoeff(const LinearExpr<T> &lhs, const LinearExpr<T> &rhs) {
    if (lhs.coeff_.size() == rhs.coeff_.size()) {
        for (size_t i = 0, iEnd = lhs.coeff_.size(); i < iEnd; i++) {
            if (lhs.coeff_[i].k_ != rhs.coeff_[i].k_) {
                return false;
            }
            if (!HashComparator()(lhs.coeff_[i].a_, rhs.coeff_[i].a_)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/**
 * Generate an expression from a LinearExpr
 *
 * This function is only applied to fundamental types. For
 * LinearExpr<Rational<T>>, see bounds.cc, because there are different rounding
 * directinos
 */
template <class T,
          typename std::enable_if_t<std::is_fundamental_v<T>> * = nullptr>
Expr lin2expr(const LinearExpr<T> &lin) {
    Expr b = makeIntConst(lin.bias_);

    for (auto &&item : lin.coeff_) {
        auto k = item.k_;
        auto a = deepCopy(item.a_);

        if (k == 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k * a.template as<IntConstNode>()->val_);
        } else if (k == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k), a);
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

template <class T>
bool operator==(const LinearExpr<T> &lhs, const LinearExpr<T> &rhs) {
    return hasIdenticalCoeff(lhs, rhs) && lhs.bias_ == rhs.bias_;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const LinearExpr<T> &lin) {
    for (auto &&[k, a] : lin.coeff_) {
        os << k << " * " << a << " + ";
    }
    os << lin.bias_;
    return os;
}

} // namespace ir

#endif // LINEAR_H
