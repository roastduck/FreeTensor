#ifndef REDUCE_OP_H
#define REDUCE_OP_H

#include <data_type.h>
#include <expr.h>

namespace ir {

enum class ReduceOp : int { Add, Mul, Min, Max };

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace ir

#endif // REDUCE_OP_H
