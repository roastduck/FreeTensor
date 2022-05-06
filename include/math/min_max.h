#ifndef FREE_TENSOR_MIN_MAX_H
#define FREE_TENSOR_MIN_MAX_H

#include <expr.h>

namespace freetensor {

/**
 * Make min(max(...), max(...), ...)
 *
 * This will remove some duplicated items in advance, to reduce the burden on
 * simplifier
 *
 * Returning nullptr means -inf
 */
Expr makeMinMax(const std::vector<std::vector<Expr>> &exprs);

/**
 * Make max(min(...), min(...), ...)
 *
 * This will remove some duplicated items in advance, to reduce the burden on
 * simplifier
 *
 * Returning nullptr means +inf
 */
Expr makeMaxMin(const std::vector<std::vector<Expr>> &exprs);

} // namespace freetensor

#endif // FREE_TENSOR_MIN_MAX_H
