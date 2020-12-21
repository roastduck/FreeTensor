#include <algorithm>

#include <pass/code_gen_c.h>

namespace ir {

void CodeGenC::visit(const VarDef &op) {
    makeIndent();
    beginBlock();

    makeIndent();
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor.shape();
    if (op->buffer_->atype() == AccessType::Cache) {
        // e.g. float x[5][5][5];
        os << gen(tensor.dtype()) << " ";
        for (auto &&dim : shape) {
            os << "[";
            (*this)(dim);
            os << "]";
        }
        os << ";" << std::endl;
    } else {
        int nthParam = params_.size();
        params_.emplace_back(op->name_);

        // e.g. const float (*restrict x)[5][5] = (float(*)[5][5])_params[0];
        if (op->buffer_->atype() == AccessType::Input) {
            os << "const ";
        }
        os << gen(tensor.dtype()) << " (*restrict ";
        os << op->name_ << ")";
        for (size_t i = 1, iEnd = shape.size(); i < iEnd; i++) { // No shape[0]
            os << "[";
            (*this)(shape[i]);
            os << "]";
        }
        os << " = (" << gen(tensor.dtype()) << "(*)";
        for (size_t i = 1, iEnd = shape.size(); i < iEnd; i++) { // No shape[0]
            os << "[";
            (*this)(shape[i]);
            os << "]";
        }
        os << ")_params[" << nthParam << "];" << std::endl;
    }

    (*this)(op->body_);
    endBlock();
}

void CodeGenC::visit(const Var &op) {
    os << op->name_;
    Visitor::visit(op);
}

void CodeGenC::visit(const Store &op) {
    makeIndent();
    if (op->indices_.empty()) {
        os << "*";
        (*this)(op->var_);
    } else {
        (*this)(op->var_);
        for (auto &&index : op->indices_) {
            os << "[";
            (*this)(index);
            os << "]";
        }
    }
    os << " = ";
    (*this)(op->expr_);
    os << ";" << std::endl;
}

void CodeGenC::visit(const Load &op) {
    if (op->indices_.empty()) {
        os << "*";
        (*this)(op->var_);
    } else {
        (*this)(op->var_);
        for (auto &&index : op->indices_) {
            os << "[";
            (*this)(index);
            os << "]";
        }
    }
}

void CodeGenC::visit(const IntConst &op) { os << std::to_string(op->val_); }

void CodeGenC::visit(const FloatConst &op) { os << std::to_string(op->val_); }

void CodeGenC::visit(const Add &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " + ";
    (*this)(op->rhs_);
    os << ")";
}

void CodeGenC::visit(const Sub &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " - ";
    (*this)(op->rhs_);
    os << ")";
}

void CodeGenC::visit(const Mul &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " * ";
    (*this)(op->rhs_);
    os << ")";
}

void CodeGenC::visit(const Div &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " / ";
    (*this)(op->rhs_);
    os << ")";
}

void CodeGenC::visit(const Mod &op) {
    os << "(";
    (*this)(op->lhs_);
    os << " + ";
    (*this)(op->rhs_);
    os << ")";
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

std::pair<std::string, std::vector<std::string>> codeGenC(const AST &op) {
    CodeGenC visitor;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = "#include <cstdint>\n"
                         "#define restrict __restrict__\n"
                         "\n"
                         "extern \"C\" {\n"
                         "\n";
    const char *tailer = "\n"
                         "}";

    return std::make_pair((std::string)header + "void run(void **_params) " +
                              visitor.toString() + tailer,
                          visitor.params());
}

} // namespace ir

