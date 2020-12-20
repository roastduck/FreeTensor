#include <pass/print_pass.h>

namespace ir {

void PrintPass::visit(const VarDef &op) {
    makeIndent();
    os << ::ir::toString(op->buffer_->atype()) << " " << op->name_ << ": ";
    auto &&tensor = op->buffer_->tensor();
    os << ::ir::toString(tensor.dtype()) << "[";
    for (size_t i = 0, iEnd = tensor.shape().size(); i < iEnd; i++) {
        os << tensor.shape()[i] << (i < iEnd - 1 ? "," : "");
    }
    os << "]" << std::endl;
    Visitor::visit(op);
}

void PrintPass::visit(const Var &op) {
    os << op->name_;
    Visitor::visit(op);
}

void PrintPass::visit(const Store &op) {
    makeIndent();
    (*this)(op->var_);
    os << "[";
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
    (*this)(op->var_);
    os << "[";
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

void PrintPass::makeIndent() {
    for (int i = 0; i < nIndent; i++) {
        os << "  ";
    }
}

std::string PrintPass::toString() { return os.str(); }

} // namespace ir

