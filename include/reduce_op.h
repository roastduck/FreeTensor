#ifndef FREE_TENSOR_REDUCE_OP_H
#define FREE_TENSOR_REDUCE_OP_H

#include <data_type.h>
#include <expr.h>

namespace freetensor {

/**
 * Operation of a `ReduceTo` node
 *
 * All operations should obey the following law: `a ?= b; a ?= c` equals to `a
 * ?= c; a ?= b`, where `?=` is any of the reduction operations
 */
enum class ReduceOp : int {
    Add,
    Sub,
    Mul,
    Min,
    Max,
    LAnd,
    LOr,
    RealDiv,
    // NOTE: FloorDiv or CeilDiv dose not obey the requried law if divisors are
    // in different signs. E.g., `603 // 625 // -492 == 0`, but `603 // -492 //
    // 625 == -1`, where `//` means FloorDiv. RoundsTowards0Div obeys the law,
    // but we don't expect it in user programs, and we only convert FloorDiv or
    // CeilDiv to it until the last lowering pass
};

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace freetensor

#endif // FREE_TENSOR_REDUCE_OP_H
