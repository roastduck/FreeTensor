#ifndef FREE_TENSOR_REDUCE_OP_H
#define FREE_TENSOR_REDUCE_OP_H

#include <data_type.h>
#include <expr.h>

namespace freetensor {

enum class ReduceOp : int { Add, Mul, Min, Max };

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace freetensor

#endif // FREE_TENSOR_REDUCE_OP_H
