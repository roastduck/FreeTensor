#ifndef FREE_TENSOR_PARALLELIZE_AS_H
#define FREE_TENSOR_PARALLELIZE_AS_H

#include <stmt.h>

namespace freetensor {

Stmt parallelizeAs(const Stmt &ast, const ID &nest, const ID &reference,
                   const ID &defId);

} // namespace freetensor

#endif // FREE_TENSOR_PARALLELIZE_AS_H
