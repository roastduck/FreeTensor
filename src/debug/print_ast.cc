#include <debug/print_ast.h>

namespace ir {

void PrintVisitor::printId(const Stmt &op) {
    if (op->id()[0] != '#') {
        os << op->id() << ":" << std::endl;
    }
}

void PrintVisitor::visit(const Any &op) {
    makeIndent();
    os << "<Any>" << std::endl;
}

void PrintVisitor::visit(const VarDef &op) {
    printId(op);
    makeIndent();
    os << ::ir::toString(op->buffer_->atype()) << " " << op->name_ << ": ";
    auto &&tensor = op->buffer_->tensor();
    os << ::ir::toString(tensor.dtype()) << "[";
    auto &&shape = tensor.shape();
    for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
        (*this)(shape[i]);
        os << (i < iEnd - 1 ? ", " : "");
    }
    os << "] ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Var &op) {
    os << op->name_;
    Visitor::visit(op);
}

void PrintVisitor::visit(const Store &op) {
    printId(op);
    makeIndent();
    os << op->var_ << "[";
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        (*this)(op->indices_[i]);
        if (i < iEnd - 1) {
            os << ", ";
        }
    }
    os << "] = ";
    (*this)(op->expr_);
    os << std::endl;
}

void PrintVisitor::visit(const Load &op) {
    os << op->var_ << "[";
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        (*this)(op->indices_[i]);
        if (i < iEnd - 1) {
            os << ", ";
        }
    }
    os << "]";
}

void PrintVisitor::visit(const AddTo &op) {
    printId(op);
    makeIndent();
    os << op->var_ << "[";
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        (*this)(op->indices_[i]);
        if (i < iEnd - 1) {
            os << ", ";
        }
    }
    os << "] += ";
    (*this)(op->expr_);
    os << std::endl;
}

void PrintVisitor::visit(const IntConst &op) { os << std::to_string(op->val_); }

void PrintVisitor::visit(const FloatConst &op) {
    os << std::to_string(op->val_);
}

void PrintVisitor::visit(const Add &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " + ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const Sub &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " - ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const Mul &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " * ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const Div &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " / ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const Mod &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " % ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const LT &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " < ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const LE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " <= ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const GT &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " > ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const GE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " >= ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const EQ &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " == ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const NE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " != ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintVisitor::visit(const Not &op) {
    os << "!";
    (*this)(op->expr_);
}

void PrintVisitor::visit(const For &op) {
    printId(op);
    makeIndent();
    os << "for " << op->iter_ << " = ";
    (*this)(op->begin_);
    os << " to ";
    (*this)(op->end_);
    os << " ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

void PrintVisitor::visit(const If &op) {
    printId(op);
    makeIndent();
    os << "if ";
    (*this)(op->cond_);
    os << " ";
    beginBlock();
    (*this)(op->thenCase_);
    endBlock();
    if (op->elseCase_.isValid()) {
        makeIndent();
        os << "else ";
        beginBlock();
        (*this)(op->elseCase_);
        endBlock();
    }
}

void PrintVisitor::visit(const Assert &op) {
    printId(op);
    makeIndent();
    os << "assert ";
    (*this)(op->cond_);
    os << " ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

std::string toString(const AST &op) {
    PrintVisitor visitor;
    visitor(op);
    return visitor.toString();
}

} // namespace ir

