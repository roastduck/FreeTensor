#ifndef SCALAR_PROP_CONST_H
#define SCALAR_PROP_CONST_H

#include <func.h>

namespace ir {

/**
 * Propagate constant scalars.
 * Scalars are values in tensors indexed with constants.
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
Stmt scalarPropConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(scalarPropConst)

} // namespace ir

#endif
