#include <debug/print_ast.h>

namespace ir {

void PrintVisitor::recur(const Expr &op) {
    if (op.isValid()) {
        (*this)(op);
    } else {
        os() << "<NULL>";
    }
}

void PrintVisitor::recur(const Stmt &op) {
    if (op.isValid()) {
        (*this)(op);
    } else {
        makeIndent();
        os() << "<NULL>" << std::endl;
    }
}

void PrintVisitor::printId(const Stmt &op) {
    if (op->hasNamedId()) {
        os() << op->id() << ":" << std::endl;
    }
}

void PrintVisitor::visit(const Any &op) {
    makeIndent();
    os() << "<Any>" << std::endl;
}

void PrintVisitor::visit(const VarDef &op) {
    printId(op);
    makeIndent();
    os() << ::ir::toString(op->buffer_->atype()) << " "
         << ::ir::toString(op->buffer_->mtype()) << " " << op->name_ << ": ";
    auto &&tensor = op->buffer_->tensor();
    os() << ::ir::toString(tensor.dtype()) << "[";
    printList(tensor.shape());
    os() << "] ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Var &op) {
    os() << op->name_;
    Visitor::visit(op);
}

void PrintVisitor::visit(const Store &op) {
    printId(op);
    makeIndent();
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "] = ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const Load &op) {
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "]";
}

void PrintVisitor::visit(const ReduceTo &op) {
    printId(op);
    if (op->atomic_) {
        makeIndent();
        os() << "// atomic" << std::endl;
    }
    makeIndent();
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "] ";
    switch (op->op_) {
    case ReduceOp::Add:
        os() << "+=";
        break;
    case ReduceOp::Min:
        os() << "min=";
        break;
    case ReduceOp::Max:
        os() << "max=";
        break;
    default:
        ASSERT(false);
    }
    os() << " ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const IntConst &op) {
    os() << std::to_string(op->val_);
}

void PrintVisitor::visit(const FloatConst &op) {
    os() << std::to_string(op->val_);
}

void PrintVisitor::visit(const Add &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " + ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Sub &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " - ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mul &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " * ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Div &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mod &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " % ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Min &op) {
    os() << "min(";
    recur(op->lhs_);
    os() << ", ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Max &op) {
    os() << "max(";
    recur(op->lhs_);
    os() << ", ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LT &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " < ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " <= ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GT &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " > ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " >= ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const EQ &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " == ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const NE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " != ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LAnd &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " && ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LOr &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " || ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LNot &op) {
    os() << "!";
    recur(op->expr_);
}

void PrintVisitor::visit(const For &op) {
    printId(op);
    if (!op->parallel_.empty()) {
        makeIndent();
        os() << "// parallel = " << op->parallel_ << std::endl;
    }
    makeIndent();
    os() << "for " << op->iter_ << " = ";
    recur(op->begin_);
    os() << " to ";
    recur(op->end_);
    os() << " ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const If &op) {
    printId(op);
    makeIndent();
    os() << "if ";
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->thenCase_);
    endBlock();
    if (op->elseCase_.isValid()) {
        makeIndent();
        os() << "else ";
        beginBlock();
        recur(op->elseCase_);
        endBlock();
    }
}

void PrintVisitor::visit(const Assert &op) {
    printId(op);
    makeIndent();
    os() << "assert ";
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Intrinsic &op) {
    os() << "intrinsic(\"";
    int i = 0;
    for (char c : op->format_) {
        if (c == '%') {
            recur(op->params_.at(i++));
        } else {
            os() << c;
        }
    }
    os() << "\")";
}

void PrintVisitor::visit(const Eval &op) {
    makeIndent();
    recur(op->expr_);
    os() << std::endl;
}

std::string toString(const AST &op) {
    PrintVisitor visitor;
    visitor(op);
    return visitor.toString(
        [](const PrintVisitor::Stream &stream) { return stream.os_.str(); });
}

} // namespace ir

