#ifndef MIN_MAX_H
#define MIN_MAX_H

#include <expr.h>

namespace ir {

/**
 * Make min(max(...), max(...), ...)
 *
 * This will remove some duplicated items in advance, to reduce the burden on
 * simplifier
 */
Expr makeMinMax(const std::vector<std::vector<Expr>> &exprs);

/**
 * Make max(min(...), min(...), ...)
 *
 * This will remove some duplicated items in advance, to reduce the burden on
 * simplifier
 */
Expr makeMaxMin(const std::vector<std::vector<Expr>> &exprs);

} // namespace ir

#endif // MIN_MAX_H
