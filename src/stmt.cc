#include <climits>
#include <cmath>

#include <stmt.h>

namespace ir {

VarDefNode::VarDefNode(const VarDefNode &other)
    : name_(other.name_), buffer_(other.buffer_.clone()),
      sizeLim_(other.sizeLim_), body_(other.body_), pinned_(other.pinned_) {}

VarDefNode &VarDefNode::operator=(const VarDefNode &other) {
    name_ = other.name_;
    buffer_ = other.buffer_.clone();
    sizeLim_ = other.sizeLim_;
    body_ = other.body_;
    pinned_ = other.pinned_;
    return *this;
}

Expr neutralVal(DataType dtype, ReduceOp op) {
    switch (dtype) {
    case DataType::Float64:
    case DataType::Float32:
        switch (op) {
        case ReduceOp::Add:
            return makeFloatConst(0.);
        case ReduceOp::Max:
            return makeFloatConst(-INFINITY);
        case ReduceOp::Min:
            return makeFloatConst(INFINITY);
        default:
            ASSERT(false);
        }

    case DataType::Int32:
        switch (op) {
        case ReduceOp::Add:
            return makeIntConst(0);
        case ReduceOp::Max:
            return makeIntConst(INT_MIN);
        case ReduceOp::Min:
            return makeIntConst(INT_MAX);
        default:
            ASSERT(false);
        }

    default:
        ASSERT(false);
    }
}

} // namespace ir
