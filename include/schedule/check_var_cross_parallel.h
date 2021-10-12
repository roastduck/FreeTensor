#ifndef CHECK_VAR_CROSS_PARALLEL_H
#define CHECK_VAR_CROSS_PARALLEL_H

#include <stmt.h>

namespace ir {

void checkVarCrossParallel(const Stmt &ast, const std::string &def,
                           MemType mtype);

}

#endif // CHECK_VAR_CROSS_PARALLEL_H
