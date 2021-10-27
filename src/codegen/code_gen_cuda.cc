#include <codegen/code_gen_cuda.h>
#include <except.h>
#include <pass/simplify.h>

#include "detail/code_gen_c.h"

namespace ir {

static std::string genCUBLASType(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "CUDA_R_32F";
    case DataType::Int32:
        return "CUDA_R_32I";
    default:
        ASSERT(false);
    }
}

void CodeGenCUDA::genAlloc(const Tensor &tensor, const std::string &rawPtr,
                           const std::string &sizePtr) {
    makeIndent();
    os() << "cudaMalloc(&" << rawPtr << ", " << sizePtr << " = ";
    for (auto &&dim : tensor.shape()) {
        (*this)(dim);
        os() << " * ";
    }
    os() << "sizeof(" << gen(tensor.dtype()) << "));" << std::endl;
}

bool CodeGenCUDA::inKernel() const {
    return streamStack_.back().name_ != "default";
}

void CodeGenCUDA::visit(const Min &op) {
    if (inKernel()) {
        os() << "min(";
        (*this)(op->lhs_);
        os() << ", ";
        (*this)(op->rhs_);
        os() << ")";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Max &op) {
    if (inKernel()) {
        os() << "max(";
        (*this)(op->lhs_);
        os() << ", ";
        (*this)(op->rhs_);
        os() << ")";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Sqrt &op) {
    os() << "runtime_sqrt("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Exp &op) {
    os() << "runtime_exp("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Abs &op) {
    os() << "runtime_abs("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Floor &op) {
    os() << "runtime_floor("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Ceil &op) {
    os() << "runtime_ceil("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const ReduceTo &op) {
    auto id = normalizeId(op->var_);
    markUseBuffer(op->var_);
    makeIndent();

    auto genAddr = [&]() {
        if (op->indices_.empty()) {
            switch (this->buffers_.at(op->var_)->mtype()) {
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
            os() << id;
            for (auto &&index : op->indices_) {
                os() << "[";
                (*this)(index);
                os() << "]";
            }
        }
    };
    auto genExpr = [&]() { (*this)(op->expr_); };

    if (op->atomic_) {
        switch (op->op_) {
        case ReduceOp::Add:
            os() << "atomicAdd(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Min:
            os() << "atomicMin(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Max:
            os() << "atomicMax(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        default:
            ASSERT(false);
        }
    } else {
        switch (op->op_) {
        case ReduceOp::Add:
            genAddr(), os() << " += ", genExpr();
            break;
        case ReduceOp::Mul:
            genAddr(), os() << " *= ", genExpr();
            break;
        case ReduceOp::Min:
            genAddr(), os() << " = min(";
            genAddr(), os() << ", ", genExpr(), os() << ")";
            break;
        case ReduceOp::Max:
            genAddr(), os() << " = max(";
            genAddr(), os() << ", ", genExpr(), os() << ")";
            break;
        default:
            ASSERT(false);
        }
        os() << ";" << std::endl;
    }
}

void CodeGenCUDA::visit(const Var &op) {
    if (op->name_ == ".threadIdx.x") {
        os() << "(int)threadIdx.x";
    } else if (op->name_ == ".threadIdx.y") {
        os() << "(int)threadIdx.y";
    } else if (op->name_ == ".threadIdx.z") {
        os() << "(int)threadIdx.z";
    } else if (op->name_ == ".blockIdx.x") {
        os() << "(int)blockIdx.x";
    } else if (op->name_ == ".blockIdx.y") {
        os() << "(int)blockIdx.y";
    } else if (op->name_ == ".blockIdx.z") {
        os() << "(int)blockIdx.z";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        if (op->property_.unroll_) {
            os() << "#pragma unroll " << op->len_ << std::endl;
        }
        CodeGenC::visit(op);
    } else if (op->property_.parallel_ == "blockIdx.x" ||
               op->property_.parallel_ == "blockIdx.y" ||
               op->property_.parallel_ == "blockIdx.z" ||
               op->property_.parallel_ == "threadIdx.x" ||
               op->property_.parallel_ == "threadIdx.y" ||
               op->property_.parallel_ == "threadIdx.z") {
        if (op->len_->nodeType() != ASTNodeType::IntConst) {
            std::ostringstream msg;
            msg << "Length of " << op->property_.parallel_
                << " should be constant, instead of " << op->len_;
            throw Error(msg.str());
        }
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_.parallel_] =
                op->len_.as<IntConstNode>()->val_;
            endBlock();
            popStream();
            Stream &stream = poppedStream_.back();
            const auto &dim = stream.threadDim_;
            auto sharedSize = stream.sharedSize_;
            auto globalSize = stream.globalSize_;

            makeIndent();
            beginBlock();
            makeIndent();
            os() << "uint8_t *__glmem = NULL;" << std::endl;
            if (globalSize > 0) {
                makeIndent();
                os() << "cudaMalloc(&__glmem, " << globalSize << ");"
                     << std::endl;
            }
            makeIndent();
            os() << "cudaFuncSetAttribute(" << kernel
                 << ", cudaFuncAttributeMaxDynamicSharedMemorySize, "
                 << std::to_string(sharedSize) << ");" << std::endl;
            makeIndent();
            os() << kernel << "<<<dim3("
                 << (dim.count("blockIdx.x") ? dim.at("blockIdx.x") : 1) << ", "
                 << (dim.count("blockIdx.y") ? dim.at("blockIdx.y") : 1) << ", "
                 << (dim.count("blockIdx.z") ? dim.at("blockIdx.z") : 1)
                 << "), dim3("
                 << (dim.count("threadIdx.x") ? dim.at("threadIdx.x") : 1)
                 << ", "
                 << (dim.count("threadIdx.y") ? dim.at("threadIdx.y") : 1)
                 << ", "
                 << (dim.count("threadIdx.z") ? dim.at("threadIdx.z") : 1)
                 << "), " << std::to_string(sharedSize) << ">>>(";
            bool first = true;
            for (auto &&[name, buffer] : stream.useBuffers_) {
                os() << (first ? "" : ", ") << normalizeId(name);
                first = false;
            }
            for (auto &&name : stream.useIters_) {
                os() << (first ? "" : ", ") << normalizeId(name);
                first = false;
            }
            os() << ", __glmem);" << std::endl;
            if (globalSize > 0) {
                makeIndent();
                os() << "cudaFree(__glmem);" << std::endl;
            }
            endBlock();
        } else {
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_.parallel_] =
                op->len_.as<IntConstNode>()->val_;
        }
    } else {
        throw Error("Unsupported parallel method " + op->property_.parallel_);
    }
}

void CodeGenCUDA::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache) {
        CodeGenC::visit(op);

    } else {
        switch (op->buffer_->mtype()) {
        case MemType::GPUGlobal: {
            markDefBuffer(op->name_, op->buffer_);

            if (inKernel()) {
                // e.g. float (*x)[5][5] = (float(*)[5][5])(__glmem + 0);
                auto &&tensor = op->buffer_->tensor();
                auto &&shape = tensor.shape();
                makeIndent();
                os() << gen(tensor.dtype()) << " (*";
                os() << normalizeId(op->name_) << ")";
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
                os() << ")(__glmem + " + std::to_string(globalStackTop_) << ");"
                     << std::endl;

                int64_t size = sizeOf(tensor.dtype());
                for (auto &&dim : shape) {
                    if (dim->nodeType() == ASTNodeType::IntConst) {
                        size *= dim.as<IntConstNode>()->val_;
                    } else {
                        throw InvalidProgram(
                            "Currently dynamic sized gpu/global "
                            "memory allocated from inside a kernel is not "
                            "supported");
                    }
                }

                streamStack_.back().globalSize_ = std::max(
                    streamStack_.back().globalSize_, globalStackTop_ + size);

                globalStackTop_ += size;
                (*this)(op->body_);
                // globalStackTop_ -= size;
                // FIXME: We have to add some sync before reusing global buffers
            } else {
                // e.g.
                // float (*x)[5][5];  // CUDA does not allow "restrict" here
                // cudaMalloc(&x, 5 * 5 * 5 * sizeof(float)); ...; cudaFree(x);
                auto &&tensor = op->buffer_->tensor();
                auto &&shape = tensor.shape();
                makeIndent();
                os() << gen(tensor.dtype()) << " (*";
                os() << normalizeId(op->name_) << ")";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    os() << "[";
                    (*this)(shape[i]);
                    os() << "]";
                }
                os() << ";" << std::endl;
                makeIndent();
                os() << "cudaMalloc(&" << normalizeId(op->name_) << ", ";
                for (auto &&dim : shape) {
                    (*this)(dim);
                    os() << " * ";
                }
                os() << "sizeof(" << gen(tensor.dtype()) << "));" << std::endl;

                (*this)(op->body_);

                makeIndent();
                os() << "cudaFree(" << normalizeId(op->name_) << ");"
                     << std::endl;
            }

            markUndefBuffer(op->name_);
            break;
        }

        case MemType::GPUShared: {
            if (!inKernel()) {
                throw InvalidProgram("Allocating a shared buffer outside a "
                                     "kernel is not allowed");
            }

            markDefBuffer(op->name_, op->buffer_);

            // A static shared memory array cannot be larger than 48KB (maybe a
            // bug of NVCC), so we allocate shared memory dynamically
            // e.g. float (*x)[5][5] = (float(*)[5][5])(__shmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor.shape();
            makeIndent();
            os() << gen(tensor.dtype()) << " (*";
            os() << normalizeId(op->name_) << ")";
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
            os() << ")(__shmem + " + std::to_string(sharedStackTop_) << ");"
                 << std::endl;

            int64_t size = sizeOf(tensor.dtype());
            for (auto &&dim : shape) {
                if (dim->nodeType() == ASTNodeType::IntConst) {
                    size *= dim.as<IntConstNode>()->val_;
                } else {
                    throw InvalidProgram("Currently dynamic sized gpu/shared "
                                         "memory is not supported");
                }
            }

            streamStack_.back().sharedSize_ = std::max(
                streamStack_.back().sharedSize_, sharedStackTop_ + size);

            sharedStackTop_ += size;
            (*this)(op->body_);
            // sharedStackTop_ -= size;
            // FIXME: We have to add some sync before reusing shared buffers

            markUndefBuffer(op->name_);
            break;
        }

        case MemType::GPULocal:
            if (!inKernel()) {
                throw InvalidProgram("Allocating a local buffer outside a "
                                     "kernel is not allowed");
            }
            CodeGenC::visit(op);
            break;

        default:
            CodeGenC::visit(op);
            break;
        }
    }
}

void CodeGenCUDA::visit(const MatMul &op) {
    if (inKernel()) {
        throw InvalidProgram("External call to a matrix multiplication from "
                             "inside a CUDA kernel is not supported");
    }

    bool transA = !op->aIsRowMajor_, transB = !op->bIsRowMajor_;
    Expr a = op->a_, b = op->b_, c = op->c_;
    Expr m = op->m_, k = op->k_, n = op->n_;
    Expr lda = op->lda_, ldb = op->ldb_, ldc = op->ldc_;
    Expr stridea = op->stridea_, strideb = op->strideb_, stridec = op->stridec_;
    if (op->cIsRowMajor_) {
        transA = !transA;
        transB = !transB;
        std::swap(transA, transB);
        std::swap(a, b);
        std::swap(lda, ldb);
        std::swap(stridea, strideb);
        std::swap(n, m);
    }

    makeIndent();
    os() << gen(dtype(op->c_)) << " _cublasAlpha = ";
    (*this)(op->alpha_);
    os() << ", _cublasBeta = ";
    (*this)(op->beta_);
    os() << ";" << std::endl;
    makeIndent();
    os() << "cublasGemmStridedBatchedEx(_ctx->cublas(), "
         << (transA ? "CUBLAS_OP_N" : "CUBLAS_OP_T") << ", "
         << (transB ? "CUBLAS_OP_N" : "CUBLAS_OP_T") << ", ";
    (*this)(m);
    os() << ", ";
    (*this)(n);
    os() << ", ";
    (*this)(k);
    os() << ", &_cublasAlpha, &";
    (*this)(a);
    os() << ", " << genCUBLASType(dtype(op->a_)) << ", ";
    (*this)(lda);
    os() << ", ";
    (*this)(stridea);
    os() << ", &";
    (*this)(b);
    os() << ", " << genCUBLASType(dtype(op->b_)) << ", ";
    (*this)(ldb);
    os() << ", ";
    (*this)(strideb);
    os() << ", &_cublasBeta, &";
    (*this)(c);
    os() << ", " << genCUBLASType(dtype(op->c_)) << ", ";
    (*this)(ldc);
    os() << ", ";
    (*this)(stridec);
    os() << ", ";
    (*this)(op->batchSize_);
    os() << ", " << genCUBLASType(dtype(op->c_)) << ", CUBLAS_GEMM_DEFAULT);"
         << std::endl;
}

std::string codeGenCUDA(const Func &func) {
    CodeGenCUDA visitor(func->params_, func->returns_);
    auto &&op = func->body_;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = R"~~~(
#include <gpu_runtime.h>

extern __shared__ uint8_t __shmem[];

extern "C" {
)~~~";
    const char *tailer = R"~~~(
}
)~~~";

    auto body = visitor.toString([&](const CodeGenCUDA::Stream &stream) {
        if (stream.name_ == "default") {
            return "void run(void **_params, void **_returns, size_t "
                   "*_retSizes, GPUContext_t _ctx) " +
                   stream.os_.str();
        } else {
            const auto &dim = stream.threadDim_;
            std::ostringstream os;
            os << "__global__ void __launch_bounds__(";
            os << (dim.count("threadIdx.x") ? dim.at("threadIdx.x") : 1);
            os << " * ";
            os << (dim.count("threadIdx.y") ? dim.at("threadIdx.y") : 1);
            os << " * ";
            os << (dim.count("threadIdx.z") ? dim.at("threadIdx.z") : 1);
            os << ") " << stream.name_ << "(";
            bool first = true;
            for (auto &&[name, buffer] : stream.useBuffers_) {
                os << (first ? "" : ", ");
                auto &&tensor = buffer->tensor();
                auto &&shape = tensor.shape();

                switch (buffer->mtype()) {
                case MemType::ByValue:
                    // e.g.
                    // __ByValArray<__ByValArray<float, 2>, 2> x;
                    for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                        os << "__ByValArray<";
                    }
                    os << CodeGenCUDA::gen(tensor.dtype());
                    for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                        ASSERT((*it)->nodeType() == ASTNodeType::IntConst);
                        os << ", " << (*it).as<IntConstNode>()->val_ << ">";
                    }
                    os << " " << visitor.normalizeId(name);
                    break;

                default:
                    // e.g. const float (*restrict x)[5][5]
                    if (buffer->atype() == AccessType::Input) {
                        os << "const ";
                    }
                    os << CodeGenCUDA::gen(tensor.dtype()) << " (*restrict ";
                    os << visitor.normalizeId(name) << ")";
                    for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                         i++) { // No shape[0]
                        ASSERT(shape[i]->nodeType() == ASTNodeType::IntConst);
                        os << "[" << shape[i].as<IntConstNode>()->val_ << "]";
                    }
                }
                first = false;
            }
            for (auto &&name : stream.useIters_) {
                os << (first ? "" : ", ") << "int "
                   << visitor.normalizeId(name);
                first = false;
            }
            os << ", uint8_t *__glmem) ";
            os << stream.os_.str() << std::endl;
            return os.str();
        }
    });
    return header + body + tailer;
}

} // namespace ir
