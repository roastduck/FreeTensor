#ifndef DETAIL_CODE_GEN_C_H
#define DETAIL_CODE_GEN_C_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <codegen/code_gen_c.h>
#include <mangle.h>

#include "code_gen.h"

namespace ir {

template <class Stream> void CodeGenC<Stream>::visit(const StmtSeq &op) {
    for (auto &&stmt : op->stmts_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            this->makeIndent();
            this->beginBlock();
            (*this)(stmt);
            this->endBlock();
        } else {
            (*this)(stmt);
        }
    }
}

template <class Stream> void CodeGenC<Stream>::visit(const VarDef &op) {
    this->markDefBuffer(op);

    this->makeIndent();
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor.shape();
    auto name = mangle(op->name_);

    if (op->buffer_->atype() == AccessType::Cache) {
        // e.g. float x[5][5][5];
        this->os() << gen(tensor.dtype()) << " " << name;
        if (op->buffer_->mtype() == MemType::GPUWarp) {
            if ((int) shape.size() && shape[0]->isConst() && shape[0].as<IntConstNode>()->val_ == 32) {
                for (int i = 1; i < (int) shape.size(); i++) {
                    this->os() << "[";
                    (*this)(shape[i]);
                    this->os() << "]";
                }
            } else {
                ERROR("GPUWarp type must have a 32-size dimension");
            }
        } else {
            for (auto &&dim : shape) {
                this->os() << "[";
                (*this)(dim);
                this->os() << "]";
            }
        }
        if (op->buffer_->mtype() == MemType::CPU) {
            // Alignment (TODO: Move to CodeGenCPU)
            bool isSingle = true;
            for (auto &&dim : shape) {
                if (dim->nodeType() != ASTNodeType::IntConst ||
                    dim.as<IntConstNode>()->val_ != 1) {
                    isSingle = false;
                    break;
                }
            }
            if (!isSingle) {
                // TODO adjust the value according to the cache line size
                this->os() << " __attribute__((aligned(64)))";
            }
        }
        this->os() << ";" << std::endl;
    } else {
        auto nthParamIter =
            std::find(params_.begin(), params_.end(), op->name_);
        auto nthReturnsIter =
            std::find_if(returns_.begin(), returns_.end(),
                         [&](const std::pair<std::string, DataType> &item) {
                             return item.first == op->name_;
                         });
        bool isParam = nthParamIter != params_.end();
        bool isReturn = nthReturnsIter != returns_.end();
        if (!isParam && !isReturn) {
            throw InvalidProgram("I/O variable " + op->name_ +
                                 " used but not defined as a function's "
                                 "parameters or return values");
        }
        std::string rawPtr;
        if (isParam) {
            int nthParam = nthParamIter - params_.begin();
            rawPtr = "_params[" + std::to_string(nthParam) + "]";
        } else {
            if (op->buffer_->atype() != AccessType::Output) {
                throw InvalidProgram(
                    "Only output variable can be as a return value");
            }
            int nthReturn = nthReturnsIter - returns_.begin();
            rawPtr = "_returns[" + std::to_string(nthReturn) + "]";
            std::string shapePtr =
                "_retShapes[" + std::to_string(nthReturn) + "]";
            std::string dimPtr = "_retDims[" + std::to_string(nthReturn) + "]";
            this->os() << "if (" + rawPtr + " == NULL) ";
            this->beginBlock();
            this->genAlloc(op->buffer_->tensor(), rawPtr, shapePtr, dimPtr);
            this->endBlock();
            this->makeIndent();
        }

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
                this->os() << gen(tensor.dtype()) << " " << name << " = *(("
                           << gen(tensor.dtype()) << "*)" << rawPtr << ");"
                           << std::endl;
            } else {
                for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                    this->os() << "__ByValArray<";
                }
                this->os() << gen(tensor.dtype());
                for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                    this->os() << ", " << (*it).as<IntConstNode>()->val_ << ">";
                }
                this->os() << " " << name << ";" << std::endl;
                std::vector<int> idx(shape.size(), 0);
                std::function<void(size_t, int)> f = [&](size_t i, int offset) {
                    if (i == shape.size()) {
                        this->makeIndent();
                        this->os() << name;
                        for (int x : idx) {
                            this->os() << "[" << x << "]";
                        }
                        this->os()
                            << " = ((" << gen(tensor.dtype()) << "*)" << rawPtr
                            << ")[" << offset << "];" << std::endl;
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
            if (shape.empty()) {
                // e.g.
                // const float &x = *((float*)_params[0]);
                if (op->buffer_->atype() == AccessType::Input) {
                    this->os() << "const ";
                }
                this->os() << gen(tensor.dtype()) << " &";
                this->os() << name << " ";
                this->os() << " = *((" << gen(tensor.dtype()) << "*)" << rawPtr
                           << ");" << std::endl;
            } else {
                // e.g.
                // const float (*restrict x)[5][5] = (float(*)[5][5])_params[0];
                if (op->buffer_->atype() == AccessType::Input) {
                    this->os() << "const ";
                }
                this->os() << gen(tensor.dtype()) << " (*restrict ";
                this->os() << name << ")";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    this->os() << "[";
                    (*this)(shape[i]);
                    this->os() << "]";
                }
                this->os() << " = (" << gen(tensor.dtype()) << "(*)";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    this->os() << "[";
                    (*this)(shape[i]);
                    this->os() << "]";
                }
                this->os() << ")" << rawPtr << ";" << std::endl;
            }
        }
    }

    (*this)(op->body_);

    this->markUndefBuffer(op);
}

template <class Stream> void CodeGenC<Stream>::visit(const Var &op) {
    this->markUseIter(op->name_);
    this->os() << mangle(op->name_);
    BaseClass::visit(op);
}

template <class Stream> void CodeGenC<Stream>::visit(const Store &op) {
    auto id = mangle(op->var_);
    this->markUseBuffer(op->var_);

    this->makeIndent();
    if (op->indices_.empty()) {
        switch (this->buffer(op->var_)->mtype()) {
        case MemType::ByValue:
        case MemType::CPU:
        case MemType::GPULocal:
            this->os() << id;
            break;
        case MemType::GPUGlobal:
        case MemType::GPUShared:
            this->os() << "*" << id;
            break;
        default:
            ASSERT(false);
        }
    } else {
        this->os() << id;
        for (auto &&index : op->indices_) {
            this->os() << "[";
            (*this)(index);
            this->os() << "]";
        }
    }
    this->os() << " = ";
    (*this)(op->expr_);
    this->os() << ";" << std::endl;
}

template <class Stream> void CodeGenC<Stream>::visit(const Load &op) {
    auto id = mangle(op->var_);
    this->markUseBuffer(op->var_);

    if (op->indices_.empty()) {
        switch (this->buffer(op->var_)->mtype()) {
        case MemType::ByValue:
        case MemType::CPU:
        case MemType::GPULocal:
            this->os() << id;
            break;
        case MemType::GPUGlobal:
        case MemType::GPUShared:
            this->os() << "*" << id;
            break;
        default:
            ASSERT(false);
        }
    } else {
        this->os() << id;
        for (auto &&index : op->indices_) {
            this->os() << "[";
            (*this)(index);
            this->os() << "]";
        }
    }
}

template <class Stream> void CodeGenC<Stream>::visit(const ReduceTo &op) {
    auto id = mangle(op->var_);
    this->markUseBuffer(op->var_);

    this->makeIndent();

    auto genAddr = [&]() {
        if (op->indices_.empty()) {
            switch (this->buffer(op->var_)->mtype()) {
            case MemType::ByValue:
            case MemType::CPU:
            case MemType::GPULocal:
                this->os() << id;
                break;
            case MemType::GPUGlobal:
            case MemType::GPUShared:
                this->os() << "*" << id;
                break;
            default:
                ASSERT(false);
            }
        } else {
            this->os() << id;
            for (auto &&index : op->indices_) {
                this->os() << "[";
                (*this)(index);
                this->os() << "]";
            }
        }
    };
    auto genExpr = [&]() { (*this)(op->expr_); };

    switch (op->op_) {
    case ReduceOp::Add:
        genAddr(), this->os() << " += ", genExpr();
        break;
    case ReduceOp::Mul:
        genAddr(), this->os() << " *= ", genExpr();
        break;
    case ReduceOp::Min:
        genAddr(), this->os()
                       << " = std::min<"
                       << this->gen(this->buffer(op->var_)->tensor().dtype())
                       << ">(";
        genAddr(), this->os() << ", ", genExpr(), this->os() << ")";
        break;
    case ReduceOp::Max:
        genAddr(), this->os()
                       << " = std::max<"
                       << this->gen(this->buffer(op->var_)->tensor().dtype())
                       << ">(";
        genAddr(), this->os() << ", ", genExpr(), this->os() << ")";
        break;
    default:
        ASSERT(false);
    }

    this->os() << ";" << std::endl;
}

template <class Stream> void CodeGenC<Stream>::visit(const IntConst &op) {
    this->os() << std::to_string(op->val_);
}

template <class Stream> void CodeGenC<Stream>::visit(const FloatConst &op) {
    if (std::isnan(op->val_)) {
        throw InvalidProgram("NaN literal in the program");
    } else if (op->val_ == INFINITY) {
        this->os() << "INFINITY";
    } else if (op->val_ == -INFINITY) {
        this->os() << "-INFINITY";
    } else {
        this->os() << std::to_string(op->val_)
                   << "f"; // FIXME: Determine the actual type
    }
}

template <class Stream> void CodeGenC<Stream>::visit(const BoolConst &op) {
    this->os() << std::to_string(op->val_);
}

template <class Stream> void CodeGenC<Stream>::visit(const Add &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " + ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Sub &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " - ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Mul &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " * ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const RealDiv &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " / ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream>
void CodeGenC<Stream>::visit(const RoundTowards0Div &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " / ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const FloorDiv &op) {
    this->os() << "floorDiv<" << this->gen(this->dtype(op)) << ">(";
    (*this)(op->lhs_);
    this->os() << ", ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const CeilDiv &op) {
    this->os() << "ceilDiv<" << this->gen(this->dtype(op)) << ">(";
    (*this)(op->lhs_);
    this->os() << ", ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Mod &op) {
    this->os() << "runtime_mod(";
    (*this)(op->lhs_);
    this->os() << ", ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Remainder &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " % ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Min &op) {
    this->os() << "std::min<" << this->gen(this->dtype(op)) << ">(";
    (*this)(op->lhs_);
    this->os() << ", ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Max &op) {
    this->os() << "std::max<" << this->gen(this->dtype(op)) << ">(";
    (*this)(op->lhs_);
    this->os() << ", ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const LT &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " < ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const LE &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " <= ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const GT &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " > ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const GE &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " >= ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const EQ &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " == ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const NE &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " != ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const LAnd &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " && ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const LOr &op) {
    this->os() << "(";
    (*this)(op->lhs_);
    this->os() << " || ";
    (*this)(op->rhs_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const LNot &op) {
    this->os() << "!";
    (*this)(op->expr_);
}

template <class Stream> void CodeGenC<Stream>::visit(const Sqrt &op) {
    this->os() << "sqrt(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Exp &op) {
    this->os() << "exp(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Square &op) {
    this->os() << "runtime_square(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Sigmoid &op) {
    this->os() << "runtime_sigmoid(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Tanh &op) {
    this->os() << "std::tanh(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Abs &op) {
    this->os() << "std::abs(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Floor &op) {
    this->os() << "std::floor(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Ceil &op) {
    this->os() << "std::ceil(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const IfExpr &op) {
    this->os() << "(";
    (*this)(op->cond_);
    this->os() << " ? ";
    (*this)(op->thenCase_);
    this->os() << " : ";
    (*this)(op->elseCase_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Cast &op) {
    this->os() << gen(op->dtype_) << "(";
    (*this)(op->expr_);
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const For &op) {
    this->markDefIter(op);
    if (op->step_->nodeType() == ASTNodeType::IntConst &&
        op->step_.as<IntConstNode>()->val_ == 1) {
        this->makeIndent();
        this->os() << "for (int " << mangle(op->iter_) << " = ";
        (*this)(op->begin_);
        this->os() << "; " << mangle(op->iter_) << " < ";
        (*this)(op->end_);
        this->os() << "; " << mangle(op->iter_) << "++) ";
        this->beginBlock();
        (*this)(op->body_);
        this->endBlock();
    } else {
        auto iterCnt = mangle(op->iter_ + ".cnt");
        this->makeIndent();
        this->os() << "for (int " << iterCnt << " = 0; " << iterCnt << " < ";
        (*this)(op->len_);
        this->os() << "; " << iterCnt << "++) ";
        this->beginBlock();
        this->makeIndent();
        this->os() << "int " << mangle(op->iter_) << " = ";
        (*this)(op->begin_);
        this->os() << " + " << iterCnt << " * ";
        (*this)(op->step_);
        this->os() << ";" << std::endl;
        (*this)(op->body_);
        this->endBlock();
    }
    this->markUndefIter(op);
}

template <class Stream> void CodeGenC<Stream>::visit(const If &op) {
    this->makeIndent();
    this->os() << "if (";
    (*this)(op->cond_);
    this->os() << ") ";
    this->beginBlock();
    (*this)(op->thenCase_);
    this->endBlock();
    if (op->elseCase_.isValid()) {
        this->makeIndent();
        this->os() << "else ";
        this->beginBlock();
        (*this)(op->elseCase_);
        this->endBlock();
    }
}

template <class Stream> void CodeGenC<Stream>::visit(const Assert &op) {
    this->makeIndent();
    this->os() << "assert(";
    (*this)(op->cond_);
    this->os() << ");" << std::endl;
    (*this)(op->body_);
}

template <class Stream> void CodeGenC<Stream>::visit(const Intrinsic &op) {
    this->os() << "(";
    int i = 0;
    for (char c : op->format_) {
        if (c == '%') {
            (*this)(op->params_.at(i++));
        } else {
            this->os() << c;
        }
    }
    this->os() << ")";
}

template <class Stream> void CodeGenC<Stream>::visit(const Eval &op) {
    this->makeIndent();
    (*this)(op->expr_);
    this->os() << ";" << std::endl;
}

template <class Stream> std::string CodeGenC<Stream>::gen(DataType dtype) {
    switch (dtype) {
    case DataType::Float64:
        return "double";
    case DataType::Float32:
        return "float";
    case DataType::Int32:
        return "int32_t";
    case DataType::Bool:
        return "bool";
    default:
        ASSERT(false);
    }
}

} // namespace ir

#endif // DETAIL_CODE_GEN_C_H
