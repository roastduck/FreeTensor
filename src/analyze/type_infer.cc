#include <analyze/type_infer.h>

namespace ir {

void TypeInfer::visitExpr(const Expr &op) {
    if (!types_.count(op)) {
        Visitor::visitExpr(op);
    }
}

#define CHK_TYPE(cond, dtype, op)                                              \
    if (!(cond)(dtype) && (dtype) != DataType::Custom) {                       \
        throw InvalidProgram("Invalid data type " + toString(dtype) + " in " + \
                             toString(op));                                    \
    }

void TypeInfer::visit(const Var &op) {
    Visitor::visit(op);
    types_[op] = DataType::Int32;
    // TODO: Able to configure this to other types
}

void TypeInfer::visit(const Load &op) {
    Visitor::visit(op);
    if (!symbolTable_.hasDef(op->var_)) {
        throw InvalidProgram("Name " + op->var_ + " used without definition");
    }
    for (auto &&idx : op->indices_) {
        CHK_TYPE(isInt, types_.at(idx), op);
    }
    types_[op] = symbolTable_.buffer(op->var_)->tensor()->dtype();
}

void TypeInfer::visit(const IntConst &op) {
    Visitor::visit(op);
    types_[op] = DataType::Int32;
    // TODO: Able to configure this to other types
}

void TypeInfer::visit(const FloatConst &op) {
    Visitor::visit(op);
    types_[op] = DataType::Float32;
    // TODO: Able to configure this to other types
}

void TypeInfer::visit(const BoolConst &op) {
    Visitor::visit(op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const Add &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Sub &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Mul &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const RealDiv &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Float32;
    // TODO: Able to configure this to other types
}

void TypeInfer::visit(const FloorDiv &op) {
    Visitor::visit(op);
    // FIXME: Currently our codegen dose not support FloorDiv on floats
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const CeilDiv &op) {
    Visitor::visit(op);
    // FIXME: Currently our codegen dose not support CeilDiv on floats
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const RoundTowards0Div &op) {
    Visitor::visit(op);
    // FIXME: Currently our codegen dose not support RoundTowards0Div on floats
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Mod &op) {
    Visitor::visit(op);
    CHK_TYPE(isInt, types_.at(op->lhs_), op);
    CHK_TYPE(isInt, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Remainder &op) {
    Visitor::visit(op);
    CHK_TYPE(isInt, types_.at(op->lhs_), op);
    CHK_TYPE(isInt, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Min &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const Max &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = upCast(types_.at(op->lhs_), types_.at(op->rhs_));
}

void TypeInfer::visit(const LT &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const LE &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const GT &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const GE &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const EQ &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const NE &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->lhs_), op);
    CHK_TYPE(isNumber, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const LAnd &op) {
    Visitor::visit(op);
    CHK_TYPE(isBool, types_.at(op->lhs_), op);
    CHK_TYPE(isBool, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const LOr &op) {
    Visitor::visit(op);
    CHK_TYPE(isBool, types_.at(op->lhs_), op);
    CHK_TYPE(isBool, types_.at(op->rhs_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const LNot &op) {
    Visitor::visit(op);
    CHK_TYPE(isBool, types_.at(op->expr_), op);
    types_[op] = DataType::Bool;
}

void TypeInfer::visit(const Sqrt &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Exp &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Square &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Sigmoid &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Tanh &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Abs &op) {
    Visitor::visit(op);
    CHK_TYPE(isNumber, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Floor &op) {
    Visitor::visit(op);
    CHK_TYPE(isFloat, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const Ceil &op) {
    Visitor::visit(op);
    CHK_TYPE(isFloat, types_.at(op->expr_), op);
    types_[op] = types_.at(op->expr_);
}

void TypeInfer::visit(const IfExpr &op) {
    Visitor::visit(op);
    CHK_TYPE(isBool, types_.at(op->cond_), op);
    types_[op] = upCast(types_.at(op->thenCase_), types_.at(op->elseCase_));
}

void TypeInfer::visit(const Cast &op) {
    Visitor::visit(op);
    types_[op] = op->dtype_;
}

void TypeInfer::visit(const Intrinsic &op) {
    Visitor::visit(op);
    types_[op] = op->retType_;
}

void TypeInfer::visit(const VarDef &op) {
    // TypeInfer borrows the symbol table from other Visitor / Mutator. Please
    // maintain the symbol table elsewhere
    ASSERT(false);
}

} // namespace ir
