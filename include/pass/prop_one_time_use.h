#ifndef FREE_TENSOR_PROP_ONE_TIME_USE_H
#define FREE_TENSOR_PROP_ONE_TIME_USE_H

#include <func.h>

namespace freetensor {

/**
 * Propagate variable if it is used only once after being assigned
 *
 * E.g. transform
 *
 * ```
 * x[0] = a
 * y[0] = x[0]
 * ```
 *
 * into
 *
 * ```
 * x[0] = a
 * y[0] = a
 * ```
 */
Stmt propOneTimeUse(const Stmt &op);

DEFINE_PASS_FOR_FUNC(propOneTimeUse)

} // namespace freetensor

#endif // FREE_TENSOR_PROP_ONE_TIME_USE_H
