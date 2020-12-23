#include <pass/print_pass.h>

namespace ir {

void PrintPass::visit(const VarDef &op) {
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

void PrintPass::visit(const Var &op) {
    os << op->name_;
    Visitor::visit(op);
}

void PrintPass::visit(const Store &op) {
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

void PrintPass::visit(const Load &op) {
    os << op->var_ << "[";
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        (*this)(op->indices_[i]);
        if (i < iEnd - 1) {
            os << ", ";
        }
    }
    os << "]";
}

void PrintPass::visit(const IntConst &op) { os << std::to_string(op->val_); }

void PrintPass::visit(const FloatConst &op) { os << std::to_string(op->val_); }

void PrintPass::visit(const Add &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " + ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const Sub &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " - ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const Mul &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " * ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const Div &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " / ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const Mod &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " % ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const LT &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " < ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const LE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " <= ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const GT &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " > ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const GE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " >= ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const EQ &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " == ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const NE &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " != ";
    (*this)(op->rhs_);
    os << ")";
}

void PrintPass::visit(const For &op) {
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

void PrintPass::visit(const If &op) {
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

std::string printPass(const AST &op) {
    PrintPass pass;
    pass(op);
    return pass.toString();
}

} // namespace ir

