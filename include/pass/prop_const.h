#ifndef PROP_CONST_H
#define PROP_CONST_H

#include <func.h>

namespace ir {

/**
 * Propagate constants and iteration variables
 *
 * E.g. transform
 *
 * ```
 * x[0] = 1
 * y[0] = x[0]
 * ```
 *
 * into
 *
 * ```
 * x[0] = 1
 * y[0] = 1
 * ```
 */
Stmt propConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(propConst)

} // namespace ir

#endif // PROP_CONST_H
