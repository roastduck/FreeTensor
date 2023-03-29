#include <data_type_infer.h>
#include <serialize/print_ast.h>

namespace freetensor {

#define CHK_TYPE(cond, dtype, op)                                              \
    if (!(cond)(dtype) && (dtype) != DataType::Custom) {                       \
        throw InvalidProgram("Invalid data type " + toString(dtype) + " in " + \
                             toString(op.self().as<ASTNode>()));               \
    }

/**
 * The return type of a math function defined in Real Domain like `exp` or
 * `tanh`, when we pass `dtype` as input data type
 */
static BaseDataType mathFuncFrom(BaseDataType dtype) {
    if (isFloat(dtype)) {
        return dtype;
    } else {
        // C++ returns double for int argument. What if for other backends? For
        // some functions, we have even not defined them for integer types in
        // CodeGen or runtime. Should we require a data type from user? (TODO)
        return DataType::Float64;
    }
}
static DataType mathFuncFrom(const DataType &dtype) {
    return mathFuncFrom(dtype.base());
}

DataType DataTypeInfer::infer(const VarNode &op) {
    // TODO: Able to configure this to other types
    return DataType::Int32;
}

DataType DataTypeInfer::infer(const LoadNode &op) { return op.loadType_; }

DataType DataTypeInfer::infer(const IntConstNode &op) {
    // TODO: Able to configure this to other types
    BaseDataType base = BaseDataType::Int32;
    SignDataType sign = SignDataType::Any;
    if (op.val_ > 0) {
        sign = SignDataType::GT0;
    } else if (op.val_ == 0) {
        sign = SignDataType::EQ0;
    } else if (op.val_ < 0) {
        sign = SignDataType::LT0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const FloatConstNode &op) {
    // TODO: Able to configure this to other types
    BaseDataType base = BaseDataType::Float32;
    SignDataType sign = SignDataType::Any;
    if (op.val_ > 0) {
        sign = SignDataType::GT0;
    } else if (op.val_ == 0) {
        sign = SignDataType::EQ0;
    } else if (op.val_ < 0) {
        sign = SignDataType::LT0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const BoolConstNode &op) {
    return DataType::Bool;
}

DataType DataTypeInfer::infer(const AddNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    BaseDataType base =
        upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        if (isLT0(op.lhs_->dtype()) || isLT0(op.rhs_->dtype())) {
            sign = SignDataType::LT0;
        } else {
            sign = SignDataType::LE0;
        }
    } else if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        if (isGT0(op.lhs_->dtype()) || isGT0(op.rhs_->dtype())) {
            sign = SignDataType::GT0;
        } else {
            sign = SignDataType::GE0;
        }
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const SubNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    BaseDataType base =
        upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        if (isLT0(op.lhs_->dtype()) || isGT0(op.rhs_->dtype())) {
            sign = SignDataType::LT0;
        } else {
            sign = SignDataType::LE0;
        }
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        if (isGT0(op.lhs_->dtype()) || isLT0(op.rhs_->dtype())) {
            sign = SignDataType::GT0;
        } else {
            sign = SignDataType::GE0;
        }
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const MulNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    BaseDataType base =
        upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    }
    if (isNE0(op.lhs_->dtype()) && isNE0(op.rhs_->dtype())) {
        if (isLE0(sign)) {
            sign = SignDataType::LT0;
        } else if (isGE0(sign)) {
            sign = SignDataType::GT0;
        } else {
            sign = SignDataType::NE0;
        }
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const RealDivNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    auto base =
        mathFuncFrom(upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base()));
    SignDataType sign = SignDataType::Any;
    if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    }
    if (isNE0(op.lhs_->dtype())) {
        if (isLE0(sign)) {
            sign = SignDataType::LT0;
        } else if (isGE0(sign)) {
            sign = SignDataType::GT0;
        } else {
            sign = SignDataType::NE0;
        }
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const FloorDivNode &op) {
    // FIXME: Currently our codegen dose not support FloorDiv on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    auto base = upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    }
    // NOTE: Even if lhs != 0, the result may still be 0
    return {base, sign};
}

DataType DataTypeInfer::infer(const CeilDivNode &op) {
    // FIXME: Currently our codegen dose not support CeilDiv on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    auto base = upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    }
    // NOTE: Even if lhs != 0, the result may still be 0
    return {base, sign};
}

DataType DataTypeInfer::infer(const RoundTowards0DivNode &op) {
    // FIXME: Currently our codegen dose not support RoundTowards0Div on floats
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    auto base = upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    }
    // NOTE: Even if lhs != 0, the result may still be 0
    return {base, sign};
}

DataType DataTypeInfer::infer(const ModNode &op) {
    CHK_TYPE(isInt, op.lhs_->dtype(), op);
    CHK_TYPE(isInt, op.rhs_->dtype(), op);
    auto base = upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    // Sign is determined by rhs
    if (isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const RemainderNode &op) {
    CHK_TYPE(isInt, op.lhs_->dtype(), op);
    CHK_TYPE(isInt, op.rhs_->dtype(), op);
    auto base = upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    // Sign is determined by lhs
    if (isGE0(op.lhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isLE0(op.lhs_->dtype())) {
        sign = SignDataType::LE0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const MinNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    BaseDataType base =
        upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isGT0(op.lhs_->dtype()) && isGT0(op.rhs_->dtype())) {
        sign = SignDataType::GT0;
    } else if (isGE0(op.lhs_->dtype()) && isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isLE0(op.lhs_->dtype()) || isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isLT0(op.lhs_->dtype()) || isLT0(op.rhs_->dtype())) {
        sign = SignDataType::LT0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const MaxNode &op) {
    CHK_TYPE(isNumber, op.lhs_->dtype(), op);
    CHK_TYPE(isNumber, op.rhs_->dtype(), op);
    BaseDataType base =
        upCast(op.lhs_->dtype().base(), op.rhs_->dtype().base());
    SignDataType sign = SignDataType::Any;
    if (isLT0(op.lhs_->dtype()) && isLT0(op.rhs_->dtype())) {
        sign = SignDataType::LT0;
    } else if (isLE0(op.lhs_->dtype()) && isLE0(op.rhs_->dtype())) {
        sign = SignDataType::LE0;
    } else if (isGE0(op.lhs_->dtype()) || isGE0(op.rhs_->dtype())) {
        sign = SignDataType::GE0;
    } else if (isGT0(op.lhs_->dtype()) || isGT0(op.rhs_->dtype())) {
        sign = SignDataType::GT0;
    }
    return {base, sign};
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
    BaseDataType base = mathFuncFrom(op.expr_->dtype().base());
    SignDataType sign = SignDataType::GE0;
    if (isNE0(op.expr_->dtype())) {
        sign = SignDataType::GT0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const ExpNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    BaseDataType base = mathFuncFrom(op.expr_->dtype().base());
    SignDataType sign = SignDataType::GT0;
    return {base, sign};
}

DataType DataTypeInfer::infer(const LnNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return mathFuncFrom(op.expr_->dtype());
}

DataType DataTypeInfer::infer(const SquareNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    BaseDataType base = mathFuncFrom(op.expr_->dtype().base());
    SignDataType sign = SignDataType::GE0;
    if (isNE0(op.expr_->dtype())) {
        sign = SignDataType::GT0;
    }
    return {base, sign};
}

DataType DataTypeInfer::infer(const SigmoidNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return mathFuncFrom(op.expr_->dtype());
}

DataType DataTypeInfer::infer(const TanhNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    return mathFuncFrom(op.expr_->dtype());
}

DataType DataTypeInfer::infer(const AbsNode &op) {
    CHK_TYPE(isNumber, op.expr_->dtype(), op);
    BaseDataType base = mathFuncFrom(op.expr_->dtype().base());
    SignDataType sign = SignDataType::GE0;
    if (isNE0(op.expr_->dtype())) {
        sign = SignDataType::GT0;
    }
    return {base, sign};
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
    // We can safely upcast both BaseDataType and SignDataType
    return upCast(op.thenCase_->dtype(), op.elseCase_->dtype());
}

DataType DataTypeInfer::infer(const CastNode &op) { return op.destType_; }

DataType DataTypeInfer::infer(const IntrinsicNode &op) { return op.retType_; }

DataType DataTypeInfer::infer(const LoadAtVersionNode &op) {
    return op.loadType_;
}

} // namespace freetensor
