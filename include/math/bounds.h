#ifndef BOUNDS_H
#define BOUNDS_H

#include <math/linear.h>
#include <math/rational.h>

namespace ir {

struct UpperBound {
    Expr expr_;
    LinearExpr<Rational<int>> lin_;

    UpperBound(const Expr &expr);
    UpperBound(const LinearExpr<Rational<int>> &lin);
};

struct LowerBound {
    Expr expr_;
    LinearExpr<Rational<int>> lin_;

    LowerBound(const Expr &expr);
    LowerBound(const LinearExpr<Rational<int>> &lin);
};

UpperBound add(const UpperBound &b1, const UpperBound &b2);
LowerBound add(const LowerBound &b1, const LowerBound &b2);

UpperBound sub(const UpperBound &b1, const LowerBound &b2);
LowerBound sub(const LowerBound &b1, const UpperBound &b2);

// we deal with multiplying constant only. Otherwise, the extreme value of
// `x * y` may not falls in the extreme value of `x` and `y`
UpperBound mul(const UpperBound &b, int k);
LowerBound mul(const LowerBound &b, int k);

// we deal with dividing by constant only. Otherwise, the extreme value of
// `x / y` may not falls in the extreme value of `x` and `y`
UpperBound floorDiv(const UpperBound &b, int k);
LowerBound floorDiv(const LowerBound &b, int k);
UpperBound ceilDiv(const UpperBound &b, int k);
LowerBound ceilDiv(const LowerBound &b, int k);

} // namespace ir

#endif // BOUNDS_H
