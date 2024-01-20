#ifndef FREE_TENSOR_VAR_SQUEEZE_H
#define FREE_TENSOR_VAR_SQUEEZE_H

#include <stmt.h>

namespace freetensor {

Stmt varSqueeze(const Stmt &ast, const ID &def, int dim);

}

#endif // FREE_TENSOR_VAR_SQUEEZE_H
