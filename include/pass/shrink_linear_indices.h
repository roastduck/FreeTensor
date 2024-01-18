#ifndef FREE_TENSOR_SHRINK_LINEAR_INDICES_H
#define FREE_TENSOR_SHRINK_LINEAR_INDICES_H

#include <stmt.h>

namespace freetensor {

/**
 * Mutator for shrinking linear indices in variables
 *
 * If a variable is consistently accessed with a linear expression, e.g., `a[8i
 * + 2j]`, and `2j` as a integer bound no larger than 8, e.g., `0 <= 2j < 4`,
 * then we can shrink the expression to be `a[4i + 2j]`.
 *
 * @{
 */
Stmt shrinkLinearIndices(const Stmt &ast, const ID &vardef);
Stmt shrinkLinearIndices(const Stmt &ast);
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_SHRINK_LINEAR_INDICES_H
