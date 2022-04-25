#include <itertools.hpp>

#include <codegen/code_gen_cuda.h>
#include <except.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

#include "detail/code_gen_c.h"

namespace ir {

static std::string genCUBLASType(DataType dtype) {
    switch (dtype) {
    case DataType::Float64:
        return "CUDA_R_64F";
    case DataType::Float32:
        return "CUDA_R_32F";
    case DataType::Int64:
        return "CUDA_R_64I";
    case DataType::Int32:
        return "CUDA_R_32I";
    default:
        ASSERT(false);
    }
}

void CodeGenCUDA::genAlloc(const Ref<Tensor> &tensor, const std::string &rawPtr,
                           const std::string &shapePtr,
                           const std::string &dimPtr) {
    auto ndim = tensor->shape().size();
    makeIndent();
    os() << shapePtr << " = " << ndim << " > 0 ? (size_t*)malloc((" << dimPtr
         << " = " << ndim << ") * sizeof(size_t)) : NULL;" << std::endl;
    makeIndent();
    os() << "checkCudaError(cudaMalloc(&" << rawPtr << ", ";
    for (auto &&[i, dim] : iter::enumerate(tensor->shape())) {
        os() << "(" << shapePtr << "[" << i << "] = ";
        (*this)(dim);
        os() << ") * ";
    }
    os() << "sizeof(" << gen(tensor->dtype()) << ")));" << std::endl;
}

void CodeGenCUDA::genScalar(const std::string &var,
                            const std::vector<Expr> &indices) {
    auto mtype = buffer(var)->mtype();
    if (indices.empty() &&
        (mtype == MemType::GPUGlobal || mtype == MemType::GPUShared)) {
        os() << "*" << mangle(var);
    } else if (!inKernel() &&
               (mtype == MemType::GPUGlobal || mtype == MemType::GPUShared ||
                mtype == MemType::GPUWarp || mtype == MemType::GPULocal)) {
        if (mtype == MemType::GPUGlobal) {
            WARNING(
                "You are accessing gpu/global memory from outside of a kernel. "
                "This is only for debugging, and it has a low performance");
            os() << "gpuScalar(";
            CodeGenC::genScalar(var, indices);
            os() << ")";
        } else {
            throw InvalidProgram("Unable to access " + ::ir::toString(mtype) +
                                 " from outside of a kernel");
        }
    } else {
        CodeGenC::genScalar(var, indices);
    }
}

bool CodeGenCUDA::inKernel() const {
    return streamStack_.back().name_ != "default";
}

void CodeGenCUDA::visitStmt(const Stmt &stmt) {
    if (streamScopes_.count(stmt)) {
        makeIndent();
        os() << "cudaStream_t __newStream;" << std::endl;
        makeIndent();
        os() << "checkCudaError(cudaStreamCreate(&__newStream));" << std::endl;
        makeIndent();
        os() << "cudaEvent_t __start, __stop;" << std::endl;
        makeIndent();
        os() << "checkCudaError(cudaEventCreate(&__start));" << std::endl;
        makeIndent();
        os() << "checkCudaError(cudaEventCreate(&__stop));" << std::endl;

        // Fork into __newStream
        makeIndent();
        os() << "checkCudaError(cudaEventRecord(__start, __stream));"
             << std::endl;
        makeIndent();
        os() << "checkCudaError(cudaStreamWaitEvent(__newStream, __start, "
                "0));"
             << std::endl;

        // Run with __newStream
        makeIndent();
        os() << "cudaStream_t __oldStream = __stream;" << std::endl;
        makeIndent();
        os() << "__stream = __newStream;" << std::endl;
        CodeGenC::visitStmt(stmt);
        makeIndent();
        os() << "__stream = __oldStream;" << std::endl;

        // Join back to __stream
        makeIndent();
        os() << "checkCudaError(cudaEventRecord(__stop, __newStream));"
             << std::endl;
        makeIndent();
        os() << "checkCudaError(cudaStreamWaitEvent(__stream, __stop, "
                "0));"
             << std::endl;

        // Destroy. In case the device is still doing work in the stream when
        // cudaStreamDestroy() is called, the function will return immediately
        // and the resources associated with the stream will be released
        // automatically once the device has completed all work in the stream
        makeIndent();
        os() << "cudaEventDestroy(__start);" << std::endl;
        makeIndent();
        os() << "cudaEventDestroy(__stop);" << std::endl;
        makeIndent();
        os() << "cudaStreamDestroy(__newStream);" << std::endl;
    } else {
        CodeGenC::visitStmt(stmt);
    }
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

void CodeGenCUDA::visit(const Tanh &op) {
    os() << "runtime_tanh("; // Defined in runtime/gpu_runtime.h
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

void CodeGenCUDA::visit(const Store &op) {
    if (buffer(op->var_)->mtype() == MemType::GPUWarp) {
        auto id = mangle(op->var_);
        markUseBuffer(op->var_);
        makeIndent();
        os() << id;
        for (int i = 1; i < (int)op->indices_.size(); i++) {
            os() << "[";
            (*this)(op->indices_[i]);
            os() << "]";
        }
        os() << " = ";
        (*this)(op->expr_);
        os() << ";" << std::endl;
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Load &op) {
    if (buffer(op->var_)->mtype() == MemType::GPUWarp) {
        auto id = mangle(op->var_);
        markUseBuffer(op->var_);
        // mask
        os() << "__shfl_sync(0x1f, ";
        // var
        os() << id;
        for (int i = 1; i < (int)op->indices_.size(); i++) {
            os() << "[";
            (*this)(op->indices_[i]);
            os() << "]";
        }
        os() << ", ";
        // srcLane
        (*this)(op->indices_[0]);
        os() << ");" << std::endl;
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const ReduceTo &op) {
    markUseBuffer(op->var_);
    makeIndent();

    auto genAddr = [&]() {
        if (this->buffer(op->var_)->mtype() == MemType::GPUWarp) {
            ASSERT(!op->indices_.empty());
            os() << mangle(op->var_);
            for (int i = 1; i < (int)op->indices_.size(); i++) {
                os() << "[";
                (*this)(op->indices_[i]);
                os() << "]";
            }
        } else {
            genScalar(op);
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
    if (op->property_->parallel_ == serialScope) {
        if (op->property_->unroll_) {
            os() << "#pragma unroll " << op->len_ << std::endl;
        }
        CodeGenC::visit(op);
    } else if (std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        if (op->len_->nodeType() != ASTNodeType::IntConst) {
            std::ostringstream msg;
            msg << "Length of " << ::ir::toString(op->property_->parallel_)
                << " should be constant, instead of " << op->len_;
            throw Error(msg.str());
        }
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            sharedStackTop_ = 0;
            globalStackTop_ = 0;
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_->parallel_] =
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
                // TODO: Use cudaMallocAsync, but it requires CUDA 11.3 and
                // device support
                os() << "checkCudaError(cudaMalloc(&__glmem, " << globalSize
                     << "));" << std::endl;
            }
            makeIndent();
            os() << "checkCudaError(cudaFuncSetAttribute(" << kernel
                 << ", cudaFuncAttributeMaxDynamicSharedMemorySize, "
                 << std::to_string(sharedSize) << "));" << std::endl;
            makeIndent();
            os() << kernel << "<<<dim3("
                 << (dim.count(blockIdxX) ? dim.at(blockIdxX) : 1) << ", "
                 << (dim.count(blockIdxY) ? dim.at(blockIdxY) : 1) << ", "
                 << (dim.count(blockIdxZ) ? dim.at(blockIdxZ) : 1) << "), dim3("
                 << (dim.count(threadIdxX) ? dim.at(threadIdxX) : 1) << ", "
                 << (dim.count(threadIdxY) ? dim.at(threadIdxY) : 1) << ", "
                 << (dim.count(threadIdxZ) ? dim.at(threadIdxZ) : 1) << "), "
                 << std::to_string(sharedSize) << ", __stream>>>(";
            bool first = true;
            for (auto &&[name, buffer] : stream.useBuffers_) {
                os() << (first ? "" : ", ") << mangle(name);
                first = false;
            }
            for (auto &&name : stream.useIters_) {
                os() << (first ? "" : ", ") << mangle(name);
                first = false;
            }
            os() << ", _params, __glmem);" << std::endl;
            if (globalSize > 0) {
                makeIndent();
                os() << "cudaFree(__glmem);" << std::endl;
            }
            endBlock();
        } else {
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_->parallel_] =
                op->len_.as<IntConstNode>()->val_;
        }
    } else if (std::holds_alternative<CUDAStreamScope>(
                   op->property_->parallel_)) {
        streamScopes_.insert(op->body_);
        CodeGenC::visit(op);
    } else {
        throw Error("Unsupported parallel method " +
                    ::ir::toString(op->property_->parallel_));
    }
}

void CodeGenCUDA::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache) {
        CodeGenC::visit(op);

    } else {
        switch (op->buffer_->mtype()) {
        case MemType::GPUGlobal: {
            markDefBuffer(op);

            if (inKernel()) {
                // e.g. float (*restirct x)[5][5] = (float(*)[5][5])(__glmem +
                // 0);
                auto &&tensor = op->buffer_->tensor();
                auto &&shape = tensor->shape();
                makeIndent();
                os() << gen(tensor->dtype()) << " (*restrict ";
                os() << mangle(op->name_) << ")";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    os() << "[";
                    (*this)(shape[i]);
                    os() << "]";
                }
                os() << " = (" << gen(tensor->dtype()) << "(*)";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    os() << "[";
                    (*this)(shape[i]);
                    os() << "]";
                }
                os() << ")(__glmem + " + std::to_string(globalStackTop_) << ");"
                     << std::endl;

                int64_t size = sizeOf(tensor->dtype());
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
                auto &&shape = tensor->shape();
                makeIndent();
                os() << gen(tensor->dtype()) << " (*";
                os() << mangle(op->name_) << ")";
                for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                     i++) { // No shape[0]
                    os() << "[";
                    (*this)(shape[i]);
                    os() << "]";
                }
                os() << ";" << std::endl;
                makeIndent();
                os() << "checkCudaError(cudaMalloc(&" << mangle(op->name_)
                     << ", ";
                for (auto &&dim : shape) {
                    (*this)(dim);
                    os() << " * ";
                }
                os() << "sizeof(" << gen(tensor->dtype()) << ")));"
                     << std::endl;

                (*this)(op->body_);

                makeIndent();
                os() << "cudaFree(" << mangle(op->name_) << ");" << std::endl;
            }

            markUndefBuffer(op);
            break;
        }

        case MemType::GPUShared: {
            if (!inKernel()) {
                throw InvalidProgram("Allocating a shared buffer outside a "
                                     "kernel is not allowed");
            }

            markDefBuffer(op);

            // A static shared memory array cannot be larger than 48KB (maybe a
            // bug of NVCC), so we allocate shared memory dynamically
            // e.g. float (*x)[5][5] = (float(*)[5][5])(__shmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            makeIndent();
            os() << gen(tensor->dtype()) << " (*";
            os() << mangle(op->name_) << ")";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << " = (" << gen(tensor->dtype()) << "(*)";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << ")(__shmem + " + std::to_string(sharedStackTop_) << ");"
                 << std::endl;

            int64_t size = sizeOf(tensor->dtype());
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

            markUndefBuffer(op);
            break;
        }

        case MemType::GPULocal:
            if (!inKernel()) {
                throw InvalidProgram("Allocating a local buffer outside a "
                                     "kernel is not allowed");
            }
            CodeGenC::visit(op);
            break;
        case MemType::GPUWarp: {
            if (!inKernel()) {
                throw InvalidProgram("Allocating a warp buffer outside a "
                                     "kernel is not allowed");
            }
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            ASSERT((int)shape.size() > 0 && shape[0]->isConst() &&
                   shape[0].as<IntConstNode>()->val_ <= 32);
            CodeGenC::visit(op);
            break;
        }
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
    beginBlock();
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
    endBlock();
}

std::string codeGenCUDA(const Func &func) {
    auto nParams = func->params_.size();

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
            std::string s =
                "void run(void **__params, void **_returns, size_t "
                "**_retShapes, size_t *_retDims, GPUContext_t _ctx) {\n";
            // We copy __params to _params, in order to pass the parameter pack
            // into a kernel
            s += "__ByValArray<void *, " + std::to_string(nParams) +
                 "> _params;\n";
            for (size_t i = 0; i < nParams; i++) {
                s += "_params[" + std::to_string(i) + "] = __params[" +
                     std::to_string(i) + "];\n";
            }
            s += "\n";

            // Set the default stream to __stream, in order to implement
            // multiple streams
            s += "cudaStream_t __stream = 0;\n";
            s += "\n";

            s += stream.os_.str();
            s += "\n";
            s += "}\n";
            return s;
        } else {
            const auto &dim = stream.threadDim_;
            std::ostringstream os;
            os << "__global__ void __launch_bounds__(";
            os << (dim.count(threadIdxX) ? dim.at(threadIdxX) : 1);
            os << " * ";
            os << (dim.count(threadIdxY) ? dim.at(threadIdxY) : 1);
            os << " * ";
            os << (dim.count(threadIdxZ) ? dim.at(threadIdxZ) : 1);
            os << ") " << stream.name_ << "(";
            bool first = true;
            for (auto &&[name, buffer] : stream.useBuffers_) {
                os << (first ? "" : ", ");
                auto &&tensor = buffer->tensor();
                auto &&shape = tensor->shape();

                switch (buffer->mtype()) {
                case MemType::ByValue:
                    // e.g.
                    // __ByValArray<__ByValArray<float, 2>, 2> x;
                    for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                        os << "__ByValArray<";
                    }
                    os << CodeGenCUDA::gen(tensor->dtype());
                    for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                        ASSERT((*it)->nodeType() == ASTNodeType::IntConst);
                        os << ", " << (*it).as<IntConstNode>()->val_ << ">";
                    }
                    os << " " << mangle(name);
                    break;

                default:
                    // e.g. const float (*restrict x)[5][5]
                    if (buffer->atype() == AccessType::Input) {
                        os << "const ";
                    }
                    os << CodeGenCUDA::gen(tensor->dtype()) << " (*restrict ";
                    os << mangle(name) << ")";
                    for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                         i++) { // No shape[0]
                        ASSERT(shape[i]->nodeType() == ASTNodeType::IntConst);
                        os << "[" << shape[i].as<IntConstNode>()->val_ << "]";
                    }
                }
                first = false;
            }
            for (auto &&name : stream.useIters_) {
                os << (first ? "" : ", ") << "int " << mangle(name);
                first = false;
            }
            os << ", __ByValArray<void *, " + std::to_string(nParams) +
                      "> _params, uint8_t *__glmem) ";
            os << stream.os_.str() << std::endl;
            return os.str();
        }
    });
    return header + body + tailer;
}

} // namespace ir
