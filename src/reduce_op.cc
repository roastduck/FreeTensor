#include <climits>
#include <cmath>

#include <reduce_op.h>

namespace freetensor {

Expr neutralVal(DataType dtype, ReduceOp op) {
    switch (dtype.base()) {
    case DataType::Float64:
    case DataType::Float32:
    case DataType::Float16:
        switch (op) {
        case ReduceOp::Add:
            return makeFloatConst(0.);
        case ReduceOp::Mul:
            return makeFloatConst(1.);
        case ReduceOp::Max:
            return makeFloatConst(-INFINITY);
        case ReduceOp::Min:
            return makeFloatConst(INFINITY);
        default:
            ASSERT(false);
        }

    case DataType::Int64:
        switch (op) {
        case ReduceOp::Add:
            return makeIntConst(0);
        case ReduceOp::Mul:
            return makeIntConst(1);
        case ReduceOp::Max:
            return makeIntConst(LLONG_MIN);
        case ReduceOp::Min:
            return makeIntConst(LLONG_MAX);
        default:
            ASSERT(false);
        }

    case DataType::Int32:
        switch (op) {
        case ReduceOp::Add:
            return makeIntConst(0);
        case ReduceOp::Mul:
            return makeIntConst(1);
        case ReduceOp::Max:
            return makeIntConst(INT_MIN);
        case ReduceOp::Min:
            return makeIntConst(INT_MAX);
        default:
            ASSERT(false);
        }

    case DataType::Bool:
        switch (op) {
        case ReduceOp::LAnd:
            return makeBoolConst(true);
        case ReduceOp::LOr:
            return makeBoolConst(false);
        default:
            ASSERT(false);
        }

    default:
        ASSERT(false);
    }
}

} // namespace freetensor
