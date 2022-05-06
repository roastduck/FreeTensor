#ifndef FREE_TENSOR_REMOVE_CYCLIC_ASSIGN_H
#define FREE_TENSOR_REMOVE_CYCLIC_ASSIGN_H

#include <pass/remove_writes.h>
namespace freetensor {

/**
 * Simplify things like `a = b; b = a`
 */
Stmt removeCyclicAssign(const Stmt &op);

DEFINE_PASS_FOR_FUNC(removeCyclicAssign)

} // namespace freetensor

#endif // FREE_TENSOR_REMOVE_CYCLIC_ASSIGN_H
