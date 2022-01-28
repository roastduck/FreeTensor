#ifndef TENSOR_PROP_CONST_H
#define TENSOR_PROP_CONST_H

#include <func.h>

namespace ir {

/**
 * Propagate constants and iteration variables
 *
 * E.g. transform
 *
 * ```
 * x[i] = 1
 * y[i] = x[i]
 * ```
 *
 * into
 *
 * ```
 * x[i] = 1
 * y[i] = 1
 * ```
 *
 * This version of const propagation is designed for both scalars and tensors.
 * For scalars, it directly invokes scalarPropConst. For tensors, it invokes the
 * Presburger solver
 */
Stmt tensorPropConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(tensorPropConst)

} // namespace ir

#endif // PROP_CONST_H
