#ifndef FREE_TENSOR_CHECK_CONFLICT_ID_H
#define FREE_TENSOR_CHECK_CONFLICT_ID_H

#include <func.h>
#include <stmt.h>

namespace freetensor {

void checkConflictId(const Stmt &ast);

inline void checkConflictId(const Func &func) { checkConflictId(func->body_); }

} // namespace freetensor

#endif
