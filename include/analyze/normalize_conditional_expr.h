#ifndef FREE_TENSOR_NORMALIZE_CONDITIONAL_EXPR_H
#define FREE_TENSOR_NORMALIZE_CONDITIONAL_EXPR_H

#include <vector>

#include <expr.h>

namespace freetensor {

/**
 * Break a expression into several conditional parts.
 *
 * This function is used for analyzing expressions with `IfExpr` inside. The
 * result will be several parts with conditions, where each part is no longer
 * with `IfExpr`.
 *
 * @param expr : The expression to be analyzed.
 * @return : A vector of pairs, where the first element is the value of the
 * expression, and the second element is the condition of the expression. The
 * condition may be null, which means the expression is always true.
 */
std::vector<std::pair<Expr /* value */, Expr /* condition, maybe null */>>
normalizeConditionalExpr(const Expr &expr);

} // namespace freetensor

#endif // FREE_TENSOR_NORMALIZE_CONDITIONAL_EXPR_H
