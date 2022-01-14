#ifndef SCALAR_PROP_H
#define SCALAR_PROP_H

#include <func.h>

namespace ir {

/**
 * Propagate constant and single-used scalars.
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
Stmt scalarProp(const Stmt &op);

DEFINE_PASS_FOR_FUNC(scalarProp)

} // namespace ir

#endif
