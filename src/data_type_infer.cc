#include <data_type_infer.h>
#include <serialize/print_ast.h>

namespace freetensor {

#define CHK_TYPE(cond, dtype, op)                                              \
    if (!(cond)(dtype) && (dtype) != DataType::Custom) {                       \
        throw InvalidProgram("Invalid data type " + toString(dtype) + " in " + \
                             toString(op.self().as<ASTNode>()));               \
    }

DataType DataTypeInfer::infer(const VarNode &op) {
    // TODO: Able to configure this to other types
    return DataType::Int32;
}

DataType DataTypeInfer::infer(const LoadNode &op) { return op.loadType_; }

DataType DataTypeInfer::infer(const IntConstNode &op) {
    // TODO: Able to configure this to other types
    return DataType::Int32;
}

DataType DataTypeInfer::infer(const FloatConstNode &op) {
    // TODO: Able to configure this to other types
    return DataType::Float32;
}

DataType DataTypeInfer::infer(const BoolConstNode &op) {
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const AddNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const SubNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const MulNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const RealDivNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Float32;
    // TODO: Able to configure this to other types
}

DataType DataTypeInfer::infer(const FloorDivNode &op) {
    // FIXME: Currently our codegen dose not support FloorDiv on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const CeilDivNode &op) {
    // FIXME: Currently our codegen dose not support CeilDiv on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const RoundTowards0DivNode &op) {
    // FIXME: Currently our codegen dose not support RoundTowards0Div on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const ModNode &op) {
    CHK_TYPE(isInt, op.lhs_->dtype(), op);
    CHK_TYPE(isInt, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const RemainderNode &op) {
    CHK_TYPE(isInt, op.lhs_->dtype(), op);
    CHK_TYPE(isInt, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const MinNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const MaxNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return upCast(op.lhs_->dtype(), op.rhs_->dtype());
}

DataType DataTypeInfer::infer(const LTNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const LENode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const GTNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const GENode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const EQNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const NENode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const LAndNode &op) {
    CHK_TYPE(isBool, op.lhs_->dtype(), op);
    CHK_TYPE(isBool, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const LOrNode &op) {
    CHK_TYPE(isBool, op.lhs_->dtype(), op);
    CHK_TYPE(isBool, op.rhs_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const LNotNode &op) {
    CHK_TYPE(isBool, op.expr_->dtype(), op);
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const SqrtNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const ExpNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const SquareNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const SigmoidNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const TanhNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const AbsNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const FloorNode &op) {
    CHK_TYPE(isFloat, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const CeilNode &op) {
    CHK_TYPE(isFloat, op.expr_->dtype(), op);
    return op.expr_->dtype();
}

DataType DataTypeInfer::infer(const IfExprNode &op) {
    CHK_TYPE(isBool, op.cond_->dtype(), op);
    return upCast(op.thenCase_->dtype(), op.elseCase_->dtype());
}

DataType DataTypeInfer::infer(const CastNode &op) { return op.destType_; }

DataType DataTypeInfer::infer(const IntrinsicNode &op) { return op.retType_; }

DataType DataTypeInfer::infer(const LoadAtVersionNode &op) {
    return op.loadType_;
}

} // namespace freetensor
