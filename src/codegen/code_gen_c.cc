#include <algorithm>
#include <cctype>
#include <functional>
#include <vector>

#include <codegen/code_gen_c.h>

namespace ir {

void CodeGenC::visit(const StmtSeq &op) {
    for (auto &&stmt : op->stmts_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            makeIndent();
            beginBlock();
            (*this)(stmt);
            endBlock();
        } else {
            (*this)(stmt);
        }
    }
}

void CodeGenC::visit(const VarDef &op) {
    markDef(normalizeId(op->name_), op->buffer_);

    makeIndent();
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor.shape();
    auto name = normalizeId(op->name_);

    if (op->buffer_->atype() == AccessType::Cache) {
        // e.g. float x[5][5][5];
        os() << gen(tensor.dtype()) << " " << name;
        for (auto &&dim : shape) {
            os() << "[";
            (*this)(dim);
            os() << "]";
        }
        os() << ";" << std::endl;
    } else {
        int nthParam = params_.size();
        params_.emplace_back(op->name_);

        switch (op->buffer_->mtype()) {
        case MemType::ByValue:
            // e.g. (1)
            // float x;
            // x = *((float*)_params[0]);

            // e.g. (2)
            // __ByValArray<__ByValArray<float, 2>, 2> x;
            // x[0][0] = *((float*)_params[0])[0];
            // x[0][1] = *((float*)_params[0])[1];
            // x[1][0] = *((float*)_params[0])[2];
            // x[1][1] = *((float*)_params[0])[3];
            if (op->buffer_->atype() != AccessType::Input) {
                throw InvalidProgram("ByValue typed var " + op->name_ +
                                     " can only be Input");
            }
            for (auto &&dim : shape) {
                if (dim->nodeType() != ASTNodeType::IntConst) {
                    throw InvalidProgram("ByValue typed var " + op->name_ +
                                         " can only have a constant size");
                }
            }
            if (shape.empty()) {
                os() << gen(tensor.dtype()) << " " << name << " = *(("
                     << gen(tensor.dtype()) << "*)_params[" << nthParam << "]);"
                     << std::endl;
            } else {
                for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                    os() << "__ByValArray<";
                }
                os() << gen(tensor.dtype());
                for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                    os() << ", " << (*it).as<IntConstNode>()->val_ << ">";
                }
                os() << " " << name << ";" << std::endl;
                std::vector<int> idx(shape.size(), 0);
                std::function<void(size_t, int)> f = [&](size_t i, int offset) {
                    if (i == shape.size()) {
                        makeIndent();
                        os() << name;
                        for (int x : idx) {
                            os() << "[" << x << "]";
                        }
                        os()
                            << " = ((" << gen(tensor.dtype()) << "*)_params["
                            << nthParam << "])[" << offset << "];" << std::endl;
                        return;
                    }
                    for (int j = 0, jEnd = shape[i].as<IntConstNode>()->val_;
                         j < jEnd; j++) {
                        idx[i] = j;
                        f(i + 1, offset * jEnd + j);
                    }
                };
                f(0, 0);
            }
            break;

        default:
            // e.g.
            // const float (*restrict x)[5][5] = (float(*)[5][5])_params[0];
            if (op->buffer_->atype() == AccessType::Input) {
                os() << "const ";
            }
            os() << gen(tensor.dtype()) << " (*restrict ";
            os() << name << ")";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << " = (" << gen(tensor.dtype()) << "(*)";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << ")_params[" << nthParam << "];" << std::endl;
        }
    }

    (*this)(op->body_);
}

void CodeGenC::visit(const Var &op) {
    os() << normalizeId(op->name_);
    CodeGen::visit(op);
}

void CodeGenC::visit(const Store &op) {
    auto id = normalizeId(op->var_);
    markUse(id);

    makeIndent();
    if (op->indices_.empty()) {
        os() << "*" << id;
    } else {
        os() << id;
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
    auto id = normalizeId(op->var_);
    markUse(id);

    if (op->indices_.empty()) {
        if (vars_.at(id).second->mtype() == MemType::ByValue) {
            os() << id;
        } else {
            os() << "*" << id;
        }
    } else {
        os() << id;
        for (auto &&index : op->indices_) {
            os() << "[";
            (*this)(index);
            os() << "]";
        }
    }
}

void CodeGenC::visit(const ReduceTo &op) {
    auto id = normalizeId(op->var_);
    markUse(id);

    makeIndent();

    auto genAddr = [&]() {
        if (op->indices_.empty()) {
            os() << "*" << id;
        } else {
            os() << id;
            for (auto &&index : op->indices_) {
                os() << "[";
                (*this)(index);
                os() << "]";
            }
        }
    };
    auto genExpr = [&]() { (*this)(op->expr_); };

    switch (op->op_) {
    case ReduceOp::Add:
        genAddr(), os() << " += ", genExpr();
        break;
    case ReduceOp::Min:
        genAddr(), os() << " = min(";
        genAddr(), os() << ", ", genExpr(), os() << ")";
        break;
    case ReduceOp::Max:
        genAddr(), os() << " = min(";
        genAddr(), os() << ", ", genExpr(), os() << ")";
        break;
    default:
        ASSERT(false);
    }

    os() << ";" << std::endl;
}

void CodeGenC::visit(const IntConst &op) { os() << std::to_string(op->val_); }

void CodeGenC::visit(const FloatConst &op) { os() << std::to_string(op->val_); }

void CodeGenC::visit(const BoolConst &op) { os() << std::to_string(op->val_); }

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

void CodeGenC::visit(const RealDiv &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const RoundTowards0Div &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " / ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const FloorDiv &op) {
    os() << "floorDiv(";
    (*this)(op->lhs_);
    os() << ", ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const CeilDiv &op) {
    os() << "ceilDiv(";
    (*this)(op->lhs_);
    os() << ", ";
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

void CodeGenC::visit(const LAnd &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " && ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const LOr &op) {
    os() << "(";
    (*this)(op->lhs_);
    os() << " || ";
    (*this)(op->rhs_);
    os() << ")";
}

void CodeGenC::visit(const LNot &op) {
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
    os() << ");" << std::endl;
    (*this)(op->body_);
}

void CodeGenC::visit(const Intrinsic &op) {
    os() << "(";
    int i = 0;
    for (char c : op->format_) {
        if (c == '%') {
            (*this)(op->params_.at(i++));
        } else {
            os() << c;
        }
    }
    os() << ")";
}

void CodeGenC::visit(const Eval &op) {
    makeIndent();
    (*this)(op->expr_);
    os() << ";" << std::endl;
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
