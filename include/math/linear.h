#ifndef LINEAR_H
#define LINEAR_H

#include <unordered_map>

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
    std::unordered_map<uint64_t, Scale<T>> coeff_;
    T bias_;
};

} // namespace ir

#endif // LINEAR_H
