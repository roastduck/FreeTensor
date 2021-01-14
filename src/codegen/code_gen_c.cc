#include <algorithm>
#include <cctype>

#include <codegen/code_gen_c.h>

namespace ir {

void CodeGenC::visit(const VarDef &op) {
    makeIndent();
    beginBlock();

    makeIndent();
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor.shape();
    if (op->buffer_->atype() == AccessType::Cache) {
        // e.g. float x[5][5][5];
        os() << gen(tensor.dtype()) << " ";
        os() << normalizeId(op->name_);
        for (auto &&dim : shape) {
            os() << "[";
            (*this)(dim);
            os() << "]";
        }
        os() << ";" << std::endl;
    } else {
        int nthParam = params_.size();
        params_.emplace_back(op->name_);

        // e.g. const float (*restrict x)[5][5] = (float(*)[5][5])_params[0];
        if (op->buffer_->atype() == AccessType::Input) {
            os() << "const ";
        }
        os() << gen(tensor.dtype()) << " (*restrict ";
        os() << normalizeId(op->name_) << ")";
        for (size_t i = 1, iEnd = shape.size(); i < iEnd; i++) { // No shape[0]
            os() << "[";
            (*this)(shape[i]);
            os() << "]";
        }
        os() << " = (" << gen(tensor.dtype()) << "(*)";
        for (size_t i = 1, iEnd = shape.size(); i < iEnd; i++) { // No shape[0]
            os() << "[";
            (*this)(shape[i]);
            os() << "]";
        }
        os() << ")_params[" << nthParam << "];" << std::endl;
    }

    (*this)(op->body_);
    endBlock();
}

void CodeGenC::visit(const Var &op) {
    os() << normalizeId(op->name_);
    Visitor::visit(op);
}

void CodeGenC::visit(const Store &op) {
    makeIndent();
    if (op->indices_.empty()) {
        os() << "*" << normalizeId(op->var_);
    } else {
        os() << normalizeId(op->var_);
        for (auto &&index : op->indices_) {
            os() << "[";
            (*this)(index);
            os() << "]";
        }
    }
    os() << " = ";
    (*this)(op->expr_);
    os() << ";" << std::endl;
}

void CodeGenC::visit(const Load &op) {
    if (op->indices_.empty()) {
        os() << "*" << normalizeId(op->var_);
    } else {
        os() << normalizeId(op->var_);
        for (auto &&index : op->indices_) {
            os() << "[";
            (*this)(index);
            os() << "]";
        }
    }
}

void CodeGenC::visit(const AddTo &op) {
    makeIndent();
    if (op->indices_.empty()) {
        os() << "*" << normalizeId(op->var_);
    } else {
        os() << normalizeId(op->var_);
        for (auto &&index : op->indices_) {
            os() << "[";
            (*this)(index);
            os() << "]";
        }
    }
    os() << " += ";
    (*this)(op->expr_);
    os() << ";" << std::endl;
}

void CodeGenC::visit(const IntConst &op) { os() << std::to_string(op->val_); }

void CodeGenC::visit(const FloatConst &op) { os() << std::to_string(op->val_); }

void CodeGenC::visit(const Add &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " + ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Sub &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " - ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Mul &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " * ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Div &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Mod &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " % ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Min &op) {
    os() << "std::min("; // TODO: Pure C?
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Max &op) {
    os() << "std::max("; // TODO: Pure C?
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const LT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " < ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const LE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " <= ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const GT &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " > ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const GE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " >= ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const EQ &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " == ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const NE &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " != ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const Not &op) {
    os() << "!";
    (*this)(op->expr_);
}

void CodeGenC::visit(const For &op) {
    makeIndent();
    os() << "for (int " << normalizeId(op->iter_) << " = ";
    (*this)(op->begin_);
    os() << "; " << normalizeId(op->iter_) << " < ";
    (*this)(op->end_);
    os() << "; " << normalizeId(op->iter_) << "++) ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

void CodeGenC::visit(const If &op) {
    makeIndent();
    os() << "if (";
    (*this)(op->cond_);
    os() << ") ";
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

void CodeGenC::visit(const Assert &op) {
    makeIndent();
    os() << "assert(";
    (*this)(op->cond_);
    os() << ") ";
    beginBlock();
    (*this)(op->body_);
    endBlock();
}

const std::string &CodeGenC::normalizeId(const std::string &old) {
    if (idCache_.count(old)) {
        return idCache_.at(old);
    }
    std::string ret = old;
    for (char &c : ret) {
        if (!isalnum(c) && c != '_') {
            c = '_';
        }
    }
    while (idFlag_.count(ret)) {
        ret += "_";
    }
    idFlag_.insert(ret);
    return idCache_[old] = ret;
}

std::string CodeGenC::gen(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "float";
    case DataType::Int32:
        return "int32_t";
    default:
        ASSERT(false);
    }
}

} // namespace ir

