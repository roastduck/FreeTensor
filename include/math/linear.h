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
