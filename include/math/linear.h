#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <map>

#include <analyze/hash.h>

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
    // Using ordered map to guarantee ASTs generated from two identical
    // `LinearExpr`s are the same
    std::map<uint64_t, Scale<T>> coeff_;
    T bias_;
};

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
        auto k = item.second.k_;
        auto a = deepCopy(item.second.a_);

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
    if (lhs.coeff_.size() != rhs.coeff_.size()) {
        return false;
    }
    for (auto &&item : lhs.coeff_) {
        if (!rhs.coeff_.count(item.first)) {
            return false;
        }
        if (item.second.k_ != rhs.coeff_.at(item.first).k_) {
            return false;
        }
    }
    if (lhs.bias_ != rhs.bias_) {
        return false;
    }
    return true;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const LinearExpr<T> &lin) {
    for (auto &&item : lin.coeff_) {
        os << item.second.k_ << " * " << item.second.a_ << " + ";
    }
    os << lin.bias_;
    return os;
}

} // namespace ir

#endif // LINEAR_H
