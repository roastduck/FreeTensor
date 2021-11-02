#ifndef REMOVE_CYCLIC_ASSIGN_H
#define REMOVE_CYCLIC_ASSIGN_H

#include <pass/remove_writes.h>
namespace ir {

/**
 * Simplify things like `a = b; b = a`
 */
Stmt removeCyclicAssign(const Stmt &op);

DEFINE_PASS_FOR_FUNC(removeCyclicAssign)

} // namespace ir

#endif // REMOVE_CYCLIC_ASSIGN_H
