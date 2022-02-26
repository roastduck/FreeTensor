#include <debug/dump_as_test.h>
#include <mangle.h>
#include <pass/undo_make_reduction.h>

#include "../codegen/detail/code_gen.h"

namespace ir {

void DumpAsTest::printId(const Stmt &op) {
    if (op->hasNamedId()) {
        makeIndent();
        os() << "ir.MarkNid(\"" << ::ir::toString(op->id()) << "\")"
             << std::endl;
    }
}

std::string DumpAsTest::asTest(DataType dtype) const {
    switch (dtype) {
    case DataType::Int32:
        return "\"int32\"";
    case DataType::Float32:
        return "\"float32\"";
    case DataType::Float64:
        return "\"float64\"";
    case DataType::Bool:
        return "\"bool\"";
    default:
        ASSERT(false);
    }
}

std::string DumpAsTest::asTest(AccessType atype) const {
    switch (atype) {
    case AccessType::Input:
        return "\"input\"";
    case AccessType::Output:
        return "\"output\"";
    case AccessType::InOut:
        return "\"inout\"";
    case AccessType::Cache:
        return "\"cache\"";
    default:
        ASSERT(false);
    }
}

std::string DumpAsTest::asTest(MemType mtype) const {
    switch (mtype) {
    case MemType::CPU:
        return "\"cpu\"";
    case MemType::GPULocal:
        return "\"gpu/local\"";
    case MemType::GPUShared:
        return "\"gpu/shared\"";
    case MemType::GPUGlobal:
        return "\"gpu/global\"";
    default:
        ASSERT(false);
    }
}

void DumpAsTest::visitStmt(const Stmt &op) {
    printId(op);
    Visitor::visitStmt(op);
}

void DumpAsTest::visit(const StmtSeq &op) {
    if (op->stmts_.empty()) {
        makeIndent();
        os() << "pass" << std::endl;
    } else {
        Visitor::visit(op);
    }
}

void DumpAsTest::visit(const VarDef &op) {
    makeIndent();
    os() << "with ir.VarDef(\"" << op->name_ << "\", (";
    auto &&tensor = op->buffer_->tensor();
    for (auto &&dim : tensor.shape()) {
        (*this)(dim);
        os() << ", ";
    }
    os() << "), " << asTest(tensor.dtype()) << ", "
         << asTest(op->buffer_->atype()) << ", " << asTest(op->buffer_->mtype())
         << ") as " << mangle(op->name_) << ": " << std::endl;
    nIndent()++;
    (*this)(op->body_);
    nIndent()--;
}

void DumpAsTest::visit(const Var &op) {
    os() << mangle(op->name_);
    Visitor::visit(op);
}

void DumpAsTest::visit(const Store &op) {
    makeIndent();
    os() << mangle(op->var_) << "[";
    if (op->indices_.empty()) {
        os() << "()";
    } else {
        printList(op->indices_);
    }
    os() << "] = ";
    (*this)(op->expr_);
    os() << std::endl;
}

void DumpAsTest::visit(const Load &op) {
    os() << mangle(op->var_) << "[";
    if (op->indices_.empty()) {
        os() << "()";
    } else {
        printList(op->indices_);
    }
    os() << "]";
}

void DumpAsTest::visit(const ReduceTo &op) { ASSERT(false); }

void DumpAsTest::visit(const IntConst &op) { os() << std::to_string(op->val_); }

void DumpAsTest::visit(const FloatConst &op) {
    os() << std::to_string(op->val_);
}

void DumpAsTest::visit(const BoolConst &op) {
    os() << (op->val_ ? "True" : "False");
}

void DumpAsTest::visit(const Add &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " + ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Sub &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " - ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Mul &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " * ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const RealDiv &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const FloorDiv &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " // ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const CeilDiv &op) {
    os() << "ir.ceil_div(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const RoundTowards0Div &op) {
    os() << "round_towards_0_div(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Mod &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " % ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Remainder &op) {
    os() << "ir.remainder(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Min &op) {
    os() << "ir.min(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const Max &op) {
    os() << "ir.max(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const LT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " < ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const LE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " <= ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const GT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " > ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const GE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " >= ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const EQ &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " == ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const NE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " != ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const LAnd &op) {
    os() << "ir.l_and(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const LOr &op) {
    os() << "ir.l_or(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void DumpAsTest::visit(const LNot &op) {
    os() << "ir.l_not(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Sqrt &op) {
    os() << "ir.sqrt(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Exp &op) {
    os() << "ir.exp(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Square &op) {
    os() << "ir.square(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Sigmoid &op) {
    os() << "ir.sigmoid(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Tanh &op) {
    os() << "ir.tanh(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Abs &op) {
    os() << "ir.abs(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Floor &op) {
    os() << "ir.floor(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const Ceil &op) {
    os() << "ir.ceil(";
    (*this)(op->expr_);
    os() << ")";
}

void DumpAsTest::visit(const IfExpr &op) {
    os() << "ir.if_then_else(";
    (*this)(op->cond_);
    os() << ", ";
    (*this)(op->thenCase_);
    os() << ", ";
    (*this)(op->elseCase_);
    os() << ")";
}

void DumpAsTest::visit(const Cast &op) {
    os() << "ir.cast(";
    (*this)(op->expr_);
    os() << ", " << asTest(op->dtype_) << ")";
}

void DumpAsTest::visit(const For &op) {
    makeIndent();
    os() << "with ir.For(\"" << op->iter_ << "\", ";
    (*this)(op->begin_);
    os() << ", ";
    (*this)(op->end_);
    os() << ") as " << mangle(op->iter_) << ":" << std::endl;
    nIndent()++;
    (*this)(op->body_);
    nIndent()--;
}

void DumpAsTest::visit(const If &op) {
    makeIndent();
    os() << "with ir.If(";
    (*this)(op->cond_);
    os() << "):";
    nIndent()++;
    (*this)(op->thenCase_);
    nIndent()--;
    if (op->elseCase_.isValid()) {
        makeIndent();
        os() << "with ir.Else():";
        nIndent()++;
        (*this)(op->elseCase_);
        nIndent()--;
    }
}

void DumpAsTest::visit(const Assert &op) {
    makeIndent();
    os() << "with ir.Assert(";
    (*this)(op->cond_);
    os() << "):";
    nIndent()++;
    (*this)(op->body_);
    nIndent()--;
}

void DumpAsTest::visit(const Intrinsic &op) {
    os() << "ir.intrinsic(\"" << op->format_ << "\"";
    for (auto &&param : op->params_) {
        os() << ", ";
        (*this)(param);
    }
    if (op->retType_ != DataType::Void) {
        os() << ", ret_type=" << asTest(op->retType_);
    }
    os() << ")";
}

void DumpAsTest::visit(const Eval &op) {
    makeIndent();
    os() << "ir.Eval(";
    (*this)(op->expr_);
    os() << ")" << std::endl;
}

std::string dumpAsTest(const Stmt &op) {
    DumpAsTest visitor;
    visitor(undoMakeReduction(op));
    return visitor.toString(
        [](const CodeGenStream &stream) { return stream.os_.str(); });
}

} // namespace ir
