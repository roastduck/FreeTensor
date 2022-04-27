#ifndef FREE_TENSOR_CHECK_VAR_CROSS_PARALLEL_H
#define FREE_TENSOR_CHECK_VAR_CROSS_PARALLEL_H

#include <stmt.h>

namespace freetensor {

void checkVarCrossParallel(const Stmt &ast, const ID &def, MemType mtype);

}

#endif // FREE_TENSOR_CHECK_VAR_CROSS_PARALLEL_H
