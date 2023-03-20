#ifndef FREE_TENSOR_REDUCE_OP_H
#define FREE_TENSOR_REDUCE_OP_H

#include <data_type.h>
#include <expr.h>

namespace freetensor {

/**
 * Operation of a `ReduceTo` node
 *
 * All operations should obey the following laws, where `?=` is any of the
 * reduction operations:
 *
 * 1. (Communicative) `a ?= b; a ?= c` equals to `a ?= c; a ?= b`.
 * 2. (Associative) `x ?= a; x ?= b; x ?= c; x ?= d` equals to `y =
 * neutral_value; y ?= a; y ?= b; z = neutral_value; z ?= c; z ?= d; x ?= y; x
 * ?= z`.
 *
 * Counter-examples:
 *
 * - `-=` or `/=` for RealDiv is not a reduction because it is against Law 2.
 * Thus, `-= x` or `/= x` will be lowered to `+= -x` or `*= 1 / x` to obey the
 * law. (NOTE: `/=` for RoundTowards0Div is also against Law 2, but it cannot be
 * lowered `*= 1 / x`, thus we cannot do parallel reduction for
 * RoundTowards0Div).
 * - `/=` for FloorDiv is not only agains Law 2, but also against Law 1. E.g.,
 * `603 // 625 // -492 == 0`, but `603 // -492 // 625 == -1`.
 */
enum class ReduceOp : int { Add, Mul, Min, Max, LAnd, LOr };

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace freetensor

#endif // FREE_TENSOR_REDUCE_OP_H
