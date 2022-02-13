#ifndef CHECK_VAR_CROSS_PARALLEL_H
#define CHECK_VAR_CROSS_PARALLEL_H

#include <stmt.h>

namespace ir {

void checkVarCrossParallel(const Stmt &ast, const ID &def, MemType mtype);

}

#endif // CHECK_VAR_CROSS_PARALLEL_H
