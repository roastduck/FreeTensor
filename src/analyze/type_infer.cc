#include <analyze/type_infer.h>

namespace ir {

void TypeInfer::visitExpr(const Expr &op,
                          const std::function<void(const Expr &)> &visitNode) {
    if (!types_.count(op)) {
        Visitor::visitExpr(op, visitNode);
    }
}

#define CHK_TYPE(cond, dtype, op)                                              \
    if (!(cond)(dtype) && (dtype) != DataType::Custom) {                       \
        throw InvalidProgram("Invalud data type " + toString(dtype) + " in " + \
                             toString(op));                                    \
    }

void TypeInfer::visit(const Var &op) {
    Visitor::visit(op);
    types_[op] = DataType::Int32;
    // TODO: Able to configure this to other types
}

void TypeInfer::visit(const Load &op) {
    Visitor::visit(op);
    if (!buffers_->count(op->var_)) {
        throw InvalidProgram("Name " + op->var_ + " used without definition");
    }
    for (auto &&idx : op->indices_) {
        CHK_TYPE(isInt, types_.at(idx), op);
    }
    types_[op] = buffers_->at(op->var_)->tensor().dtype();
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

void TypeInfer::visit(const Cast &op) {
    Visitor::visit(op);
    types_[op] = op->dtype_;
}

void TypeInfer::visit(const Intrinsic &op) {
    Visitor::visit(op);
    types_[op] = op->retType_;
}

void TypeInfer::visit(const VarDef &op) {
    if (buffers_->count(op->name_)) {
        throw InvalidProgram("Nested VarDef with the same name is not allowed");
    }
    (*buffers_)[op->name_] = op->buffer_;
    Visitor::visit(op);
    buffers_->erase(op->name_);
}

} // namespace ir

