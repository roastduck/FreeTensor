#ifndef FREE_TENSOR_CHECK_CONFLICT_ID_H
#define FREE_TENSOR_CHECK_CONFLICT_ID_H

#include <visitor.h>

namespace freetensor {

void checkConflictId(const Stmt &ast);

DEFINE_PASS_FOR_FUNC(checkConflictId)

} // namespace freetensor

#endif
