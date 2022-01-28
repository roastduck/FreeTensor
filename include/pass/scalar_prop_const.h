#ifndef SCALAR_PROP_CONST_H
#define SCALAR_PROP_CONST_H

#include <func.h>

namespace ir {

/**
 * Propagate scalars of constant value or only depending on iteration variables.
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
 *
 * This version of const propagation is designed for only scalars and meant to
 * be fast. It uses traditional dataflow techniques
 */
Stmt scalarPropConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(scalarPropConst)

} // namespace ir

#endif
