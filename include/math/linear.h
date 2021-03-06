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
std::ostream &operator<<(std::ostream &os, const LinearExpr<T> &lin) {
    for (auto &&item : lin.coeff_) {
        os << item.second.k_ << " * " << item.second.a_ << " + ";
    }
    os << lin.bias_;
    return os;
}

} // namespace ir

#endif // LINEAR_H
