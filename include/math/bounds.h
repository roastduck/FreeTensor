#ifndef BOUNDS_H
#define BOUNDS_H

#include <math/linear.h>
#include <math/rational.h>

namespace ir {

class UpperBound {
    Expr expr_;
    LinearExpr<Rational<int>> lin_;

  public:
    UpperBound(const Expr &expr)
        : expr_(expr), lin_{{{getHash(expr), {1, deepCopy(expr)}}}, 0} {}
    UpperBound(const LinearExpr<Rational<int>> &lin) : lin_(lin) {}
    UpperBound(LinearExpr<Rational<int>> &&lin) : lin_(std::move(lin)) {}

    const Expr &expr();
    const LinearExpr<Rational<int>> &lin() const { return lin_; }
};

class LowerBound {
    Expr expr_;
    LinearExpr<Rational<int>> lin_;

  public:
    LowerBound(const Expr &expr)
        : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}
    LowerBound(const LinearExpr<Rational<int>> &lin) : lin_(lin) {}
    LowerBound(LinearExpr<Rational<int>> &&lin) : lin_(std::move(lin)) {}

    const Expr &expr();
    const LinearExpr<Rational<int>> &lin() const { return lin_; }
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
