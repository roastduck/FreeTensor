#ifndef FREE_TENSOR_BOUNDS_H
#define FREE_TENSOR_BOUNDS_H

#include <iostream>
#include <optional>
#include <unordered_set>

#include <math/linear.h>
#include <math/rational.h>

namespace freetensor {

namespace detail {
ASTNodeType reverseCmp(ASTNodeType type);
};

class UpperBound {
    Expr expr_;
    std::optional<std::unordered_set<std::string>> allNames_;
    LinearExpr<Rational<int64_t>> lin_;

  public:
    UpperBound(const Expr &expr)
        : expr_(expr), lin_{{{1, deepCopy(expr)}}, 0} {}
    UpperBound(const LinearExpr<Rational<int64_t>> &lin) : lin_(lin) {}
    UpperBound(LinearExpr<Rational<int64_t>> &&lin) : lin_(std::move(lin)) {}

    const Expr &expr();
    const std::unordered_set<std::string> &allNames();
    const LinearExpr<Rational<int64_t>> &lin() const { return lin_; }

    friend std::ostream &operator<<(std::ostream &os, const UpperBound &b) {
        return os << b.lin();
    }
};

class LowerBound {
    Expr expr_;
    std::optional<std::unordered_set<std::string>> allNames_;
    LinearExpr<Rational<int64_t>> lin_;

  public:
    LowerBound(const Expr &expr) : expr_(expr), lin_{{{1, expr}}, 0} {}
    LowerBound(const LinearExpr<Rational<int64_t>> &lin) : lin_(lin) {}
    LowerBound(LinearExpr<Rational<int64_t>> &&lin) : lin_(std::move(lin)) {}

    const Expr &expr();
    const std::unordered_set<std::string> &allNames();
    const LinearExpr<Rational<int64_t>> &lin() const { return lin_; }

    friend std::ostream &operator<<(std::ostream &os, const LowerBound &u) {
        return os << u.lin();
    }
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

bool alwaysLT(const UpperBound &b1, const LowerBound &b2);
bool alwaysLE(const UpperBound &b1, const LowerBound &b2);

/**
 * Dirive the bound of "x" in a "LINEAR(x) OP 0"
 *
 * E.g. convert "2 * x + 3 * y > 0" to x > -3/2 * y"
 */
template <class T>
std::pair<std::optional<LowerBound>, std::optional<UpperBound>>
lin2bounds(const LinearExpr<T> &_lin, ASTNodeType cmp, const Expr &x) {
    typedef std::pair<std::optional<LowerBound>, std::optional<UpperBound>>
        RetType;

    // 1. Remove x from lin
    // 2. Convert to a rational linear because we need to do division later
    LinearExpr<Rational<int64_t>> lin;
    std::optional<Rational<int64_t>> selfK;
    lin.bias_ = _lin.bias_;
    lin.coeff_.reserve(_lin.coeff_.size() - 1);
    for (auto &&[k, a] : _lin.coeff_) {
        if (HashComparator()(a, x)) {
            ASSERT(!selfK.has_value());
            selfK = std::make_optional<Rational<int64_t>>(k);
        } else {
            lin.coeff_.emplace_back(Scale<Rational<int64_t>>{k, a});
        }
    }
    if (!selfK.has_value() || *selfK == 0) {
        return RetType(std::nullopt, std::nullopt);
    }

    // 3. Normalize according to selfK
    // Now x is at the left side and the other items are at the right side
    if (*selfK < 0) {
        cmp = detail::reverseCmp(cmp);
    }
    lin.bias_ /= -*selfK;
    for (auto &item : lin.coeff_) {
        item.k_ /= -*selfK;
    }

    // 4. Construct the bounds according to cmp
    // We normalize LT and GT to LE and GE according to the following:
    // selfK * x < y <==> selfK * x <= y - 1  (as an integer expression)
    // x < 1/selfK * y <==> x <= 1/selfK * y - 1/selfK
    switch (cmp) {
    case ASTNodeType::LE:
        return RetType(std::nullopt, std::make_optional<UpperBound>(lin));
    case ASTNodeType::LT:
        return RetType(
            std::nullopt,
            std::make_optional<UpperBound>(LinearExpr<Rational<int64_t>>{
                lin.coeff_, lin.bias_ - 1 / std::abs(*selfK)}));
    case ASTNodeType::GE:
        return RetType(std::make_optional<LowerBound>(lin), std::nullopt);
    case ASTNodeType::GT:
        return RetType(
            std::make_optional<LowerBound>(LinearExpr<Rational<int64_t>>{
                lin.coeff_, lin.bias_ + 1 / std::abs(*selfK)}),
            std::nullopt);
    case ASTNodeType::EQ:
        return RetType(std::make_optional<LowerBound>(lin),
                       std::make_optional<UpperBound>(lin));
    default:
        return RetType(std::nullopt, std::nullopt);
    }
}

namespace detail {

inline ASTNodeType reverseCmp(ASTNodeType type) {
    switch (type) {
    case ASTNodeType::LT:
        return ASTNodeType::GT;
    case ASTNodeType::LE:
        return ASTNodeType::GE;
    case ASTNodeType::GT:
        return ASTNodeType::LT;
    case ASTNodeType::GE:
        return ASTNodeType::LE;
    case ASTNodeType::EQ:
        return ASTNodeType::EQ;
    case ASTNodeType::NE:
        return ASTNodeType::NE;
    default:
        ASSERT(false);
    }
}

}; // namespace detail

} // namespace freetensor

#endif // FREE_TENSOR_BOUNDS_H
