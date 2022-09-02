#include <itertools.hpp>

#include <codegen/code_gen_cuda.h>
#include <except.h>
#include <math/utils.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

#include "detail/code_gen_c.h"

namespace freetensor {

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

void CodeGenCUDA::genScalar(const VarDef &def,
                            const std::vector<Expr> &indices) {
    auto &&var = def->name_;
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
            CodeGenC::genScalar(def, indices);
            os() << ")";
        } else {
            throw InvalidProgram("Unable to access " +
                                 ::freetensor::toString(mtype) +
                                 " from outside of a kernel");
        }
    } else if (def->buffer_->mtype() == MemType::GPULocal ||
               def->buffer_->mtype() == MemType::GPUWarp) {
        // Likely registers, no wrapping inside a mdspan
        os() << mangle(def->name_);
        for (auto &&index : indices) {
            os() << "[";
            (*this)(index);
            os() << "]";
        }
    } else {
        CodeGenC::genScalar(def, indices);
    }
}

bool CodeGenCUDA::inKernel() const {
    return streamStack_.back().name_ != "default" || inCublas_;
}

void CodeGenCUDA::exprOr1(const std::unordered_map<ParallelScope, Expr> &dict,
                          const ParallelScope &key) {
    if (dict.count(key)) {
        (*this)(dict.at(key));
    } else {
        os() << 1;
    }
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

void CodeGenCUDA::visit(const Alloc &op) {
    auto &&buf = buffer(op->var_);
    auto &&tensor = buf->tensor();
    auto &&shape = tensor->shape();
    auto &&dtype = tensor->dtype();
    ASSERT(buf->mtype() == MemType::GPUGlobalHeap);

    // e.g.
    // x_opt = mdspan_r<int, extents<5, 5>>(cudaNew(5 * 5 * sizeof(int)));
    os() << mangle(op->var_) << "_opt = ";
    genMdPtrDef(buf, [&]() {
        os() << "cudaNew(";
        for (auto &&dim : shape) {
            (*this)(dim);
            os() << " * ";
        }
        os() << "sizeof(" << gen(dtype) << "))";
    });
    os() << ";" << std::endl;
}

void CodeGenCUDA::visit(const Free &op) {
    ASSERT(buffer(op->var_)->mtype() == MemType::GPUGlobalHeap);

    // e.g. auto x_ptr = x.data_handle();
    //      x_opt.drop();
    //      x_opt = std::nullopt;
    //      cudaFree(x_ptr);
    auto &&name = mangle(op->var_);
    makeIndent();
    os() << "auto " << name << "_ptr = " << name << ".data_handle();"
         << std::endl;
    makeIndent();
    os() << name << "_opt.drop();" << std::endl;
    makeIndent();
    os() << name << "_opt = std::nullopt;" << std::endl;
    makeIndent();
    os() << "cudaFree(" << name << "_ptr);" << std::endl;
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
        case ReduceOp::Sub:
            os() << "atomicSub(&", genAddr(), os() << ", ", genExpr();
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
        case ReduceOp::LAnd:
            os() << "atomicAnd(&", genAddr(), os() << ", (bool)(", genExpr();
            os() << "));" << std::endl;
            break;
        case ReduceOp::LOr:
            os() << "atomicOr(&", genAddr(), os() << ", (bool)(", genExpr();
            os() << "));" << std::endl;
            break;
        default:
            ASSERT(false);
        }
    } else {
        switch (op->op_) {
        case ReduceOp::Add:
            genAddr(), os() << " += ", genExpr();
            break;
        case ReduceOp::Sub:
            genAddr(), os() << " -= ", genExpr();
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
        case ReduceOp::LAnd:
            genAddr(), this->os() << " &= (bool)(", genExpr(),
                this->os() << ")";
            break;
        case ReduceOp::LOr:
            genAddr(), this->os() << " |= (bool)(", genExpr(),
                this->os() << ")";
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
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            sharedStackTop_ = 0;
            auto oldGlobalStackTop = globalStackTop_;
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_->parallel_] = op->len_;
            endBlock();
            globalStackTop_ =
                oldGlobalStackTop; // Because we are not reducing
                                   // globalStackTop_ inside a kernel
            popStream();
            Stream &stream = poppedStream_.back();
            const auto &dim = stream.threadDim_;
            auto sharedSize = stream.sharedSize_;
            if (sharedSize > 96 * 1024) {
                throw InvalidProgram("Shared memory too large");
            }

            makeIndent();
            os() << "checkCudaError(cudaFuncSetAttribute(" << kernel
                 << ", cudaFuncAttributeMaxDynamicSharedMemorySize, "
                 << std::to_string(sharedSize) << "));" << std::endl;
            makeIndent();
            os() << kernel << "<<<dim3(";
            exprOr1(dim, blockIdxX);
            os() << ", ";
            exprOr1(dim, blockIdxY);
            os() << ", ";
            exprOr1(dim, blockIdxZ);
            os() << "), dim3(";
            exprOr1(dim, threadIdxX);
            os() << ", ";
            exprOr1(dim, threadIdxY);
            os() << ", ";
            exprOr1(dim, threadIdxZ);
            os() << "), " << std::to_string(sharedSize) << ", __stream>>>(";
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
        } else {
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->property_->parallel_] = op->len_;
        }
    } else if (std::holds_alternative<CUDAStreamScope>(
                   op->property_->parallel_)) {
        streamScopes_.insert(op->body_);
        CodeGenC::visit(op);
    } else {
        throw Error("Unsupported parallel method " +
                    ::freetensor::toString(op->property_->parallel_));
    }
}

void CodeGenCUDA::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache) {
        CodeGenC::visit(op);

    } else {
        switch (op->buffer_->mtype()) {
        case MemType::GPUGlobal: {

            // e.g. auto x = mdspan_r<float, extents<5, 5>>(__glmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            makeIndent();
            os() << "auto " << mangle(op->name_) << " = ";
            genMdPtrDef(op->buffer_,
                        "__glmem + " + std::to_string(globalStackTop_));
            os() << ";" << std::endl;

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

            // Align to 128 bytes (TODO: look up cache line size from Target)
            size = ceilDiv<int64_t>(size, 128) * 128;

            globalSize_ = std::max(globalSize_, globalStackTop_ + size);

            globalStackTop_ += size;
            markDefBuffer(op);
            (*this)(op->body_);
            if (inKernel()) {
                // globalStackTop_ -= size;
                // FIXME: We have to add some sync before reusing global buffers
            } else {
                globalStackTop_ -= size;
            }
            markUndefBuffer(op);
            break;
        }

        case MemType::GPUGlobalHeap: {
            if (inKernel()) {
                throw InvalidProgram("gpu/global/heap memory allocated from "
                                     "inside a kernel is not supported");
            } else {
                // e.g. UncheckedOpt<mdspan_r<float, std::extents<5, 5>>> x_opt;
                //      auto &x = *x_opt;
                auto &&name = mangle(op->name_);
                makeIndent();
                os() << "UncheckedOpt<";
                genMdPtrType(op->buffer_);
                os() << "> " << name << "_opt;" << std::endl;
                makeIndent();
                os() << "auto &" << name << " = *" << name << "_opt;"
                     << std::endl;

                markDefBuffer(op);
                (*this)(op->body_);
                markUndefBuffer(op);
            }
            break;
        }

        case MemType::GPUShared: {
            if (!inKernel()) {
                throw InvalidProgram("Allocating a shared buffer outside a "
                                     "kernel is not allowed");
            }

            // A static shared memory array cannot be larger than 48KB (maybe a
            // bug of NVCC), so we allocate shared memory dynamically
            // auto x = mdspan<float, extents<5, 5>>(__shmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            makeIndent();
            os() << "auto " << mangle(op->name_) << " = ";
            genMdPtrDef(op->buffer_,
                        "__shmem + " + std::to_string(sharedStackTop_));
            os() << ";" << std::endl;

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

            markDefBuffer(op);
            sharedStackTop_ += size;
            (*this)(op->body_);
            // sharedStackTop_ -= size;
            // FIXME: We have to add some sync before reusing shared buffers
            markUndefBuffer(op);
            break;
        }

        case MemType::GPULocal:
        case MemType::GPUWarp: {
            if (!inKernel()) {
                throw InvalidProgram(
                    "Allocating a " +
                    ::freetensor::toString(op->buffer_->mtype()) +
                    " buffer outside a "
                    "kernel is not allowed");
            }
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            // e.g. float x[5][5][5];
            makeIndent();
            os() << gen(tensor->dtype()) << " " << mangle(op->name_);
            if (op->buffer_->mtype() == MemType::GPUWarp) {
                ASSERT((int)shape.size() > 0 && shape[0]->isConst() &&
                       shape[0].as<IntConstNode>()->val_ <= 32);
                if (!shape.empty() && shape[0]->isConst() &&
                    shape[0].as<IntConstNode>()->val_ == 32) {
                    for (size_t i = 1; i < shape.size(); i++) {
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
            os() << ";" << std::endl;

            markDefBuffer(op);
            (*this)(op->body_);
            markUndefBuffer(op);
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

    inCublas_ = true;

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
    os() << gen(op->c_->dtype()) << " _cublasAlpha = ";
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
    os() << ", " << genCUBLASType(op->a_->dtype()) << ", ";
    (*this)(lda);
    os() << ", ";
    (*this)(stridea);
    os() << ", &";
    (*this)(b);
    os() << ", " << genCUBLASType(op->b_->dtype()) << ", ";
    (*this)(ldb);
    os() << ", ";
    (*this)(strideb);
    os() << ", &_cublasBeta, &";
    (*this)(c);
    os() << ", " << genCUBLASType(op->c_->dtype()) << ", ";
    (*this)(ldc);
    os() << ", ";
    (*this)(stridec);
    os() << ", ";
    (*this)(op->batchSize_);
    os() << ", " << genCUBLASType(op->c_->dtype()) << ", CUBLAS_GEMM_DEFAULT);"
         << std::endl;
    endBlock();

    inCublas_ = false;
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

            // Allocate stack for gpu/global
            s += "uint8_t *__glmem = (uint8_t*)cudaNew(" +
                 std::to_string(visitor.globalSize()) + ");\n";
            s += "\n";

            s += stream.os_.str();
            s += "\n";

            // Free stack for gpu/global
            s += "cudaFree(__glmem);\n";

            s += "}\n";
            return s;
        } else {
            const auto &dim = stream.threadDim_;
            std::ostringstream os;
            os << "__global__ void ";
            for (auto &&[d, len] : dim) {
                if (len.isValid() && len->nodeType() != ASTNodeType::IntConst) {
                    goto dynamic_dim;
                }
            }
            os << "__launch_bounds__(";
            os << (dim.count(threadIdxX)
                       ? dim.at(threadIdxX).as<IntConstNode>()->val_
                       : 1);
            os << " * ";
            os << (dim.count(threadIdxY)
                       ? dim.at(threadIdxY).as<IntConstNode>()->val_
                       : 1);
            os << " * ";
            os << (dim.count(threadIdxZ)
                       ? dim.at(threadIdxZ).as<IntConstNode>()->val_
                       : 1);
            os << ") ";
        dynamic_dim:
            os << stream.name_ << "(";
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
                    // e.g. mdspan<float, extents<5, 5>> x
                    visitor.genMdPtrType(os, buffer,
                                         buffer->atype() == AccessType::Input);
                    os << " " << mangle(name);
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

} // namespace freetensor
