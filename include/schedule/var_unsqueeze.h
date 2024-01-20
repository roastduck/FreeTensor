#ifndef FREE_TENSOR_VAR_UNSQUEEZE_H
#define FREE_TENSOR_VAR_UNSQUEEZE_H

#include <stmt.h>

namespace freetensor {

Stmt varUnsqueeze(const Stmt &ast, const ID &def, int dim);

}

#endif // FREE_TENSOR_VAR_UNSQUEEZE_H
