#ifndef PROP_ONE_TIME_USE_H
#define PROP_ONE_TIME_USE_H

#include <func.h>

namespace ir {

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

} // namespace ir

#endif // PROP_ONE_TIME_USE_H
