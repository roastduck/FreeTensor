#ifndef FREE_TENSOR_REDUCE_OP_H
#define FREE_TENSOR_REDUCE_OP_H

#include <data_type.h>
#include <expr.h>

namespace freetensor {

/**
 * Operation of a `ReduceTo` node
 *
 * All operations should obey the commutative law. Note that although `-` does
 * not obey the commutative law, `-=` does
 *
 * Here is no support for `/=` because of the lack of support from OpenMP and
 * CUDA
 */
enum class ReduceOp : int { Add, Sub, Mul, Min, Max, LAnd, LOr };

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace freetensor

#endif // FREE_TENSOR_REDUCE_OP_H
