#ifndef FREE_TENSOR_TENSOR_PROP_CONST_H
#define FREE_TENSOR_TENSOR_PROP_CONST_H

#include <func.h>

namespace freetensor {

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
 *
 * @param subAST : If set, only propagate in this sub-tree
 */
Stmt tensorPropConst(const Stmt &op, const ID &subAST = ID());

DEFINE_PASS_FOR_FUNC(tensorPropConst)

} // namespace freetensor

#endif // FREE_TENSOR_PROP_CONST_H
