#include <debug/print_ast.h>

namespace ir {

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
    if (op->info_acc_lower_.isValid() && op->info_acc_len_.isValid()) {
        makeIndent();
        os() << "// lower = [";
        printList(*op->info_acc_lower_);
        os() << "], len = [";
        printList(*op->info_acc_len_);
        os() << "]" << std::endl;
    }
    makeIndent();
    os() << ::ir::toString(op->buffer_->atype()) << " "
         << ::ir::toString(op->buffer_->mtype()) << " " << op->name_ << ": ";
    auto &&tensor = op->buffer_->tensor();
    os() << ::ir::toString(tensor.dtype()) << "[";
    printList(tensor.shape());
    os() << "] ";
    beginBlock();
    (*this)(op->body_);
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
    (*this)(op->expr_);
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
    (*this)(op->expr_);
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
    (*this)(op->lhs_);
    os() << " + ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Sub &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " - ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mul &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " * ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Div &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mod &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " % ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Min &op) {
    os() << "min(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Max &op) {
    os() << "max(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " < ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " <= ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " > ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " >= ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const EQ &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " == ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const NE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " != ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LAnd &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " && ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LOr &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " || ";
    (*this)(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LNot &op) {
    os() << "!";
    (*this)(op->expr_);
}

void PrintVisitor::visit(const For &op) {
    printId(op);
    if (op->info_max_begin_.isValid() && op->info_min_end_.isValid()) {
        makeIndent();
        os() << "// max_begin = ";
        (*this)(op->info_max_begin_);
        os() << ", min_end = ";
        (*this)(op->info_min_end_);
        os() << std::endl;
    }
    if (!op->parallel_.empty()) {
        makeIndent();
        os() << "// parallel = " << op->parallel_ << std::endl;
    }
    makeIndent();
    os() << "for " << op->iter_ << " = ";
    (*this)(op->begin_);
    os() << " to ";
    (*this)(op->end_);
    os() << " ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

void PrintVisitor::visit(const If &op) {
    printId(op);
    makeIndent();
    os() << "if ";
    (*this)(op->cond_);
    os() << " ";
    beginBlock();
    (*this)(op->thenCase_);
    endBlock();
    if (op->elseCase_.isValid()) {
        makeIndent();
        os() << "else ";
        beginBlock();
        (*this)(op->elseCase_);
        endBlock();
    }
}

void PrintVisitor::visit(const Assert &op) {
    printId(op);
    makeIndent();
    os() << "assert ";
    (*this)(op->cond_);
    os() << " ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Intrinsic &op) {
    os() << "intrinsic(\"";
    int i = 0;
    for (char c : op->format_) {
        if (c == '%') {
            (*this)(op->params_.at(i++));
        } else {
            os() << c;
        }
    }
    os() << "\")";
}

void PrintVisitor::visit(const Eval &op) {
    makeIndent();
    (*this)(op->expr_);
    os() << std::endl;
}

std::string toString(const AST &op) {
    PrintVisitor visitor;
    visitor(op);
    return visitor.toString(
        [](const PrintVisitor::Stream &stream) { return stream.os_.str(); });
}

} // namespace ir

