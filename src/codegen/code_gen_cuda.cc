#ifdef FT_WITH_CUDA

#include <analyze/all_uses.h>
#include <analyze/find_stmt.h>
#include <codegen/code_gen_cuda.h>
#include <config.h>
#include <container_utils.h>
#include <except.h>
#include <math/utils.h>
#include <pass/const_fold.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

#include "detail/code_gen_c.h"

namespace freetensor {

static std::string genCUBLASType(const DataType &dtype) {
    switch (dtype.base()) {
    case DataType::Float64:
        return "CUDA_R_64F";
    case DataType::Float32:
        return "CUDA_R_32F";
    case DataType::Float16:
        return "CUDA_R_16F";
    case DataType::Int64:
        return "CUDA_R_64I";
    case DataType::Int32:
        return "CUDA_R_32I";
    default:
        ASSERT(false);
    }
}

static std::string genCUTLASSType(const DataType &dtype) {
    switch (dtype.base()) {
    case DataType::Float64:
        return "double";
    case DataType::Float32:
        return "float";
    case DataType::Float16:
        return "cutlass::half_t";
    case DataType::Int64:
        return "int64_t";
    case DataType::Int32:
        return "int32_t";
    case DataType::Bool:
        return "bool";
    default:
        ASSERT(false);
    }
}

static bool canUseTensorCore(const Ref<GPUTarget> &target, DataType dtypeA,
                             DataType dtypeB, DataType dtypeC) {
    if (target->computeCapability().first >= 7 && dtypeA == DataType::Float16 &&
        dtypeB == DataType::Float16 &&
        (dtypeC == DataType::Float16 || dtypeC == DataType::Float32)) {
        return true;
    }
    if (target->computeCapability().first >= 8 && dtypeA == DataType::Float64 &&
        dtypeB == DataType::Float64 && dtypeC == DataType::Float64) {
        return true;
    }
    return false;
}

std::function<std::ostream &(std::ostream &)>
CodeGenCUDA::genMdPtrType(const VarDef &def, bool isConst) {
    Ref<Buffer> buf = def->buffer_;
    if (buf->tensor()->shape().empty() &&
        (buf->mtype() == MemType::GPUGlobal ||
         buf->mtype() == MemType::GPUGlobalHeap)) {
        // Use pointer instead of reference for scalars, because when passing an
        // argument from host to a kernel, a reference means copy the value from
        // CPU to GPU, while a pointer means passing the address

        // NOTE: `[=]` implicitly capturing `this` is deprecated in C++20. Using
        // `[=]` will trigger a warning in GCC (because of deprecation), but
        // using
        // `[=, this]` will trigger a warning in Clang<17 (because it will think
        // `this` is duplicated).
#if defined(__clang__) && __clang_major__ < 17
        return [=](std::ostream &os) -> std::ostream & {
#else
        return [=, this](std::ostream &os) -> std::ostream & {
#endif
            if (isConst) {
                os << "const ";
            }
            return os << gen(buf->tensor()->dtype()) << " *";
        };
    }
    return CodeGenC<CodeGenCUDAStream>::genMdPtrType(def, isConst);
}

void CodeGenCUDA::genMdPtrDef(const VarDef &def,
                              const std::function<void()> &genRawPtr,
                              bool isConst) {
    auto &&buf = def->buffer_;
    if (buf->tensor()->shape().empty() &&
        (buf->mtype() == MemType::GPUGlobal ||
         buf->mtype() == MemType::GPUGlobalHeap)) {
        // Use pointer instead of reference for scalars, because when passing an
        // argument from host to a kernel, a reference means copy the value from
        // CPU to GPU, while a pointer means passing the address
        this->os() << "((" << genMdPtrType(def, isConst) << ")(";
        genRawPtr();
        this->os() << "))";
        return;
    }
    CodeGenC<CodeGenCUDAStream>::genMdPtrDef(def, genRawPtr, isConst);
}

std::string CodeGenCUDA::gen(const DataType &dtype) {
    if (inKernel() && dtype.base() == DataType::Float16) {
        return "__half";
    } else {
        return CodeGenC::gen(dtype);
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
    // cudaNew is defined in gpu_runtime.h. This allocation is for return
    // values, so we don't allocate from our pool
    os() << rawPtr << " = cudaNew(";
    for (auto &&[i, dim] : views::enumerate(tensor->shape())) {
        os() << "(" << shapePtr << "[" << i << "] = ";
        (*this)(dim);
        os() << ") * ";
    }
    os() << "sizeof(" << gen(tensor->dtype()) << "), __stream);" << std::endl;
}

void CodeGenCUDA::genScalar(const VarDef &def,
                            const std::vector<Expr> &indices) {
    auto &&var = def->name_;
    auto mtype = buffer(var)->mtype();
    if (!inKernel() &&
        (mtype == MemType::GPUGlobal || mtype == MemType::GPUGlobalHeap ||
         mtype == MemType::GPUShared || mtype == MemType::GPUWarp ||
         mtype == MemType::GPULocal)) {
        if (mtype == MemType::GPUGlobal || mtype == MemType::GPUGlobalHeap) {
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
    } else if (inKernel() &&
               (mtype == MemType::CPU || mtype == MemType::CPUHeap)) {
        throw InvalidProgram("Unable to access " +
                             ::freetensor::toString(mtype) +
                             " from inside a kernel");
    } else if (indices.empty() && (mtype == MemType::GPUGlobal ||
                                   mtype == MemType::GPUGlobalHeap)) {
        os() << "*" << mangle(var);
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
    return streamStack_.back().name_ != "default" || inMatmul_;
}

void CodeGenCUDA::exprOr1(const std::unordered_map<ParallelScope, Expr> &dict,
                          const ParallelScope &key) {
    if (dict.count(key)) {
        (*this)(dict.at(key));
    } else {
        os() << 1;
    }
}

void CodeGenCUDA::enterKernel(const Stmt &body) {
    std::string kernel = kernelPrefix_ + "_kernel" + std::to_string(nKernel_++);
    pushStream(kernel);
    sharedStackTop_ = makeIntConst(0);
    auto oldGlobalStackTop = globalStackTop_;
    beginBlock();
    (*this)(body);
    endBlock();
    globalStackTop_ = oldGlobalStackTop; // Because we are not reducing
                                         // globalStackTop_ inside a kernel
    popStream();

    Stream &stream = poppedStream_.back();
    const auto &dim = stream.threadDim_;
    auto sharedSize = stream.sharedSize_;

    makeIndent();
    os() << "checkCudaError(cudaFuncSetAttribute(" << kernel
         << ", cudaFuncAttributeMaxDynamicSharedMemorySize, ";
    (*this)(sharedSize);
    os() << "));" << std::endl;
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
    os() << "), ";
    (*this)(sharedSize);
    os() << ", __stream>>>(";
    bool first = true;
    for (auto &&[name, d] : stream.useDefs_) {
        os() << (first ? "" : ", ") << mangle(name);
        first = false;
    }
    for (auto &&name : stream.useIters_) {
        os() << (first ? "" : ", ") << mangle(name);
        first = false;
    }
    os() << ", params, __glmem);" << std::endl;

    // While run time error inside a kernel can be checked in future
    // synchronizations, invalid kernel launches has to be checked here
    makeIndent();
    os() << "checkCudaError(cudaGetLastError());" << std::endl;

    if (Config::debugCUDAWithUM()) {
        makeIndent();
        os() << "checkCudaError(cudaStreamSynchronize(__stream));" << std::endl;
    }
}

bool CodeGenCUDA::canRunInKernel(const Stmt &stmt) {
    if (!findAllStmt(
             stmt,
             "!(<For>|<If>|<Assert>|<Assume>|<StmtSeq>|<Store>|<ReduceTo>)")
             .empty()) {
        // No VarDef here, because memory are more likely to be wasted
        // (allocated in a more conservative way) inside a kernel due to lack of
        // synchronization
        return false;
    }

    for (auto &&_loop : findAllStmt(stmt, "<For>")) {
        auto &&loop = _loop.as<ForNode>();
        // If some inner loops is already parallelized, we can't safely extend
        // the kernel scope, otherwise a barrier at the end of each kernel
        // launch is ignored. This branch also reject OpenMP scopes for (maybe
        // future) hybrid CPU-GPU parallelization.
        if (loop->property_->parallel_ != serialScope) {
            return false;
        }
    }

    for (auto &&var : allUses(stmt)) {
        auto mtype = buffer(var)->mtype();
        if (mtype != MemType::GPULocal && mtype != MemType::GPUWarp &&
            mtype != MemType::GPUShared && mtype != MemType::GPUGlobal &&
            mtype != MemType::GPUGlobalHeap) {
            return false;
        }
    }

    return true;
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

void CodeGenCUDA::visit(const Ln &op) {
    os() << "runtime_log("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Sin &op) {
    os() << "runtime_sin("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Cos &op) {
    os() << "runtime_cos("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Tan &op) {
    os() << "runtime_tan("; // Defined in runtime/gpu_runtime.h
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

void CodeGenCUDA::visit(const Cast &op) {
    if (op->destType_.base() == DataType::Float16) {
        switch (op->expr_->dtype().base()) {
        case DataType::Int32:
            os() << "__int2half_rn(";
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Int64:
            os() << "__ll2half_rn(";
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Float16:
            (*this)(op->expr_);
            break;
        case DataType::Float32:
            os() << "__float2half_rn(";
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Float64:
            os() << "__double2half("; // Always `_rn` (round to nearest even)
            (*this)(op->expr_);
            os() << ")";
            break;
        default:
            throw InvalidProgram("Converting from " +
                                 freetensor::toString(op->dtype()) +
                                 " to float16 is not supported");
        }
    } else if (op->expr_->dtype().base() == DataType::Float16) {
        switch (op->destType_.base()) {
        case DataType::Int32:
            os() << "__half2int_rn(";
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Int64:
            os() << "__half2ll_rn(";
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Float16:
            (*this)(op->expr_);
            break;
        case DataType::Float32:
            os() << "__half2float("; // Short to long, no rounding is needed
            (*this)(op->expr_);
            os() << ")";
            break;
        case DataType::Float64:
            os() << "__half2double("; // Short to long, no rounding is needed
            (*this)(op->expr_);
            os() << ")";
            break;
        default:
            throw InvalidProgram("Converting from float16 to " +
                                 freetensor::toString(op->dtype()) +
                                 " is not supported");
        }
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Store &op) {
    if (buffer(op->var_)->mtype() == MemType::GPUWarp) {
        auto id = mangle(op->var_);
        markUse(op->var_);
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
        markUse(op->var_);
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
    auto &&vardef = def(op->var_);
    auto &&buf = vardef->buffer_;
    auto &&tensor = vardef->buffer_->tensor();
    auto &&shape = tensor->shape();
    auto &&dtype = tensor->dtype();
    ASSERT(buf->mtype() == MemType::GPUGlobalHeap);

    // e.g.
    // x_opt = mdspan_r<int, extents<5, 5>>(cudaNewFromPool(5 * 5 * sizeof(int),
    // __stream, ctx->gpuGlobalDyanmicPool()));
    makeIndent();
    os() << mangle(op->var_) << "_opt = ";
    genMdPtrDef(vardef, [&]() {
        os() << "cudaNewFromPool(";
        for (auto &&dim : shape) {
            (*this)(dim);
            os() << " * ";
        }
        os() << "sizeof(" << gen(dtype)
             << "), __stream, ctx->gpuGlobalDynamicPool())";
    });
    os() << ";" << std::endl;
}

void CodeGenCUDA::visit(const Free &op) {
    ASSERT(buffer(op->var_)->mtype() == MemType::GPUGlobalHeap);

    // e.g. auto x_ptr = x.data_handle();
    //      x_opt.drop();
    //      x_opt = std::nullopt;
    //      cudaFreeAsync(x_ptr, __stream);
    auto &&name = mangle(op->var_);
    makeIndent();
    os() << "auto " << name << "_ptr = " << name << ".data_handle();"
         << std::endl;
    makeIndent();
    os() << name << "_opt.drop();" << std::endl;
    makeIndent();
    os() << name << "_opt = std::nullopt;" << std::endl;
    makeIndent();
    os() << "cudaFreeAsync(" << name << "_ptr, __stream);" << std::endl;
}

void CodeGenCUDA::visit(const ReduceTo &op) {
    markUse(op->var_);
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

    if (op->sync_) {
        switch (op->op_) {
        case ReduceOp::Add:
            os() << "atomicAdd(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Min:
            // Defined in `runtime/gpu_runtime.h`
            os() << "runtimeAtomicMin(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Max:
            // Defined in `runtime/gpu_runtime.h`
            os() << "runtimeAtomicMax(&", genAddr(), os() << ", ", genExpr();
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

        // The followings are not supported by CUDA's atomic functions, do
        // atomic CAS by ourselves. `atomicUpdate` is defined in
        // `runtime/gpu_runtime.h`
        case ReduceOp::Mul:
            makeIndent();
            os() << "atomicUpdate(";
            genScalar(op);
            // User names are prefixed by an `_`, so we are safe with `x` here
            os() << ", [&](" << gen(buffer(op->var_)->tensor()->dtype())
                 << " x) { return x * (";
            (*this)(op->expr_);
            os() << "); });" << std::endl;
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
            CodeGenC::visit(op);
        } else if (!inKernel()) {
            if (canRunInKernel(op)) {
                enterKernel(op);
            } else {
                CodeGenC::visit(op);
            }
        } else {
            CodeGenC::visit(op);
        }
    } else if (std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        if (!inKernel()) {
            enterKernel(op);
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
    if (isInputting(op->buffer_->atype()) ||
        isOutputting(op->buffer_->atype()) || op->viewOf_.has_value()) {
        CodeGenC::visit(op);

    } else {
        switch (op->buffer_->mtype()) {
        case MemType::GPUGlobal: {

            // e.g. auto &&x = mdspan_r<float, extents<5, 5>>(__glmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            makeIndent();
            os() << "auto &&" << mangle(op->name_) << " = ";
            genMdPtrDef(op, [this]() {
                os() << "__glmem + (";
                (*this)(globalStackTop_);
                os() << ")";
            });
            os() << ";" << std::endl;

            Expr size = makeIntConst(sizeOf(tensor->dtype()));
            for (auto &&dim : shape) {
                size = makeMul(size, dim);
            }

            // Align to 128 bytes (TODO: look up cache line size from Target)
            size = makeMul(makeCeilDiv(size, makeIntConst(128)),
                           makeIntConst(128));

            globalSize_ =
                constFold(makeMax(globalSize_, makeAdd(globalStackTop_, size)));

            auto oldGlobalStackTop = globalStackTop_;
            globalStackTop_ = constFold(makeAdd(globalStackTop_, size));
            markDef(op);
            (*this)(op->body_);
            if (inKernel()) {
                // globalStackTop_ = oldGlobalStackTop;
                // FIXME: We have to add some sync before reusing global buffers
            } else {
                globalStackTop_ = oldGlobalStackTop;
            }
            markUndef(op);
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
                os() << "UncheckedOpt<" << genMdPtrType(op) << "> " << name
                     << "_opt;" << std::endl;
                makeIndent();
                os() << "auto &" << name << " = *" << name << "_opt;"
                     << std::endl;

                markDef(op);
                (*this)(op->body_);
                markUndef(op);
            }
            break;
        }

        case MemType::GPUShared: {
            if (!inKernel()) {
                enterKernel(op);
                return;
            }

            // A static shared memory array cannot be larger than 48KB (maybe a
            // bug of NVCC), so we allocate shared memory dynamically
            // auto &&x = mdspan<float, extents<5, 5>>(__shmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor->shape();
            makeIndent();
            os() << "auto &&" << mangle(op->name_) << " = ";
            genMdPtrDef(op, [this]() {
                os() << "__shmem + (";
                (*this)(sharedStackTop_);
                os() << ")";
            });
            os() << ";" << std::endl;

            Expr size = makeIntConst(sizeOf(tensor->dtype()));
            for (auto &&dim : shape) {
                size = makeMul(size, dim);
            }

            streamStack_.back().sharedSize_ =
                constFold(makeMax(streamStack_.back().sharedSize_,
                                  makeAdd(sharedStackTop_, size)));

            markDef(op);
            sharedStackTop_ = constFold(makeAdd(sharedStackTop_, size));
            (*this)(op->body_);
            // FIXME: Restore sharedStackTop_, but we have to add some sync
            // before reusing shared buffers
            markUndef(op);
            break;
        }

        case MemType::GPULocal:
        case MemType::GPUWarp: {
            if (!inKernel()) {
                enterKernel(op);
                return;
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

            markDef(op);
            (*this)(op->body_);
            markUndef(op);
            break;
        }

        default:
            CodeGenC::visit(op);
            break;
        }
    }
}

void CodeGenCUDA::visit(const MatMul &op) {
    bool thisOpInKernel = inKernel();
    inMatmul_ = true;

    bool transA = !op->aIsRowMajor_, transB = !op->bIsRowMajor_,
         transC = !op->cIsRowMajor_;
    Expr a = op->a_, b = op->b_, c = op->c_;
    Expr m = op->m_, k = op->k_, n = op->n_;
    Expr lda = op->lda_, ldb = op->ldb_, ldc = op->ldc_;
    Expr stridea = op->stridea_, strideb = op->strideb_, stridec = op->stridec_;

    switch (op->backend_) {
    case MatMulBackend::Cublas: {
        if (thisOpInKernel) {
            throw InvalidProgram("External call to a matrix multiplication "
                                 "implemented by cuBLAS from inside a CUDA "
                                 "kernel is not supported");
        }

        if (op->cIsRowMajor_) {
            transA = !transA;
            transB = !transB;
            transC = false;
            std::swap(transA, transB);
            std::swap(a, b);
            std::swap(lda, ldb);
            std::swap(stridea, strideb);
            std::swap(n, m);
        }

        makeIndent();
        beginBlock();
        makeIndent();
        os() << gen(op->c_->dtype()) << " cublasAlpha = ";
        (*this)(op->alpha_);
        os() << ", cublasBeta = ";
        (*this)(op->beta_);
        os() << ";" << std::endl;
        makeIndent();
        os() << "cublasGemmStridedBatchedEx(ctx->cublas(), "
             << (transA ? "CUBLAS_OP_N" : "CUBLAS_OP_T") << ", "
             << (transB ? "CUBLAS_OP_N" : "CUBLAS_OP_T") << ", ";
        (*this)(m);
        os() << ", ";
        (*this)(n);
        os() << ", ";
        (*this)(k);
        os() << ", &cublasAlpha, &";
        (*this)(a);
        os() << ", " << genCUBLASType(a->dtype()) << ", ";
        (*this)(lda);
        os() << ", ";
        (*this)(stridea);
        os() << ", &";
        (*this)(b);
        os() << ", " << genCUBLASType(b->dtype()) << ", ";
        (*this)(ldb);
        os() << ", ";
        (*this)(strideb);
        os() << ", &cublasBeta, &";
        (*this)(c);
        os() << ", " << genCUBLASType(c->dtype()) << ", ";
        (*this)(ldc);
        os() << ", ";
        (*this)(stridec);
        os() << ", ";
        (*this)(op->batchSize_);
        os() << ", " << genCUBLASType(c->dtype()) << ", CUBLAS_GEMM_DEFAULT);"
             << std::endl;
        endBlock();
        break;
    }

    case MatMulBackend::Cutlass: {
        if (thisOpInKernel) {
            throw InvalidProgram("External call to a matrix multiplication "
                                 "implemented by CUTLASS from inside a CUDA "
                                 "kernel is not supported");
        }

        makeIndent();
        beginBlock();
        makeIndent();
        os() << "using Gemm = cutlass::gemm::device::Gemm<"
             << genCUTLASSType(a->dtype()) << ", "
             << (transA ? "cutlass::layout::ColumnMajor"
                        : "cutlass::layout::RowMajor")
             << ", " << genCUTLASSType(b->dtype()) << ", "
             << (transB ? "cutlass::layout::ColumnMajor"
                        : "cutlass::layout::RowMajor")
             << ", " << genCUTLASSType(c->dtype()) << ", "
             << (transC ? "cutlass::layout::ColumnMajor"
                        : "cutlass::layout::RowMajor")
             << ", " << genCUTLASSType(c->dtype()) // TODO: accumulator type
             << ", "
             << (canUseTensorCore(target_, a->dtype(), b->dtype(), c->dtype())
                     ? "cutlass::arch::OpClassTensorOp"
                     : "cutlass::arch::OpClassSimt")
             << ", FT_CUTLASS_ARCH>;" << std::endl;
        makeIndent();
        os() << "Gemm gemm;" << std::endl;
        // In order for clearer error message, please keep the explicit argument
        // types in the following statement.
        makeIndent();
        os() << "checkCutlassError(gemm(Gemm::Arguments{{";
        (*this)(m);
        os() << ", ";
        (*this)(n);
        os() << ", ";
        (*this)(k);
        os() << "}, Gemm::TensorRefA{(const " << genCUTLASSType(a->dtype())
             << "*)&";
        (*this)(a);
        os() << ", ";
        (*this)(lda);
        os() << "}, Gemm::TensorRefB{(const " << genCUTLASSType(b->dtype())
             << "*)&";
        (*this)(b);
        os() << ", ";
        (*this)(ldb);
        os() << "}, Gemm::TensorRefC{(const " << genCUTLASSType(c->dtype())
             << "*)&";
        (*this)(c);
        os() << ", ";
        (*this)(ldc);
        os() << "}, Gemm::TensorRefD{(" << genCUTLASSType(c->dtype()) << "*)&";
        (*this)(c);
        os() << ", ";
        (*this)(ldc);
        os() << "}, Gemm::EpilogueOutputOp::Params{("
             << genCUTLASSType(c->dtype()) << ")(";
        (*this)(op->alpha_);
        os() << "), (" << genCUTLASSType(c->dtype()) << ")(";
        (*this)(op->beta_);
        os() << ")}}, nullptr, __stream));" << std::endl;
        endBlock();
        break;
    }

    case MatMulBackend::CutlassMicroThread: {
        if (!thisOpInKernel) {
            throw InvalidProgram(
                "A MatMul's micro kernel can only be called inside a kernel");
        }

        ASSERT(op->cIsRowMajor_);

        auto &&prop = op->cutlassMicroKernelProperty_;

        makeIndent();
        os() << "matmul_thread<";
        (*this)(m);
        os() << ", ";
        (*this)(n);
        os() << ", ";
        (*this)(k);
        os() << ", " << prop->nWarpBatch_ << ", " << prop->nWarpM_ << ", "
             << prop->nWarpN_ << ", " << (transA ? "true" : "false") << ", "
             << (transB ? "true" : "false") << ", "
             << genCUTLASSType(a->dtype()) << ", " << genCUTLASSType(b->dtype())
             << ", " << genCUTLASSType(c->dtype()) << ">((const "
             << genCUTLASSType(a->dtype()) << "*)&(";
        (*this)(a);
        os() << "), (const " << genCUTLASSType(b->dtype()) << "*)&(";
        (*this)(b);
        os() << "), (" << genCUTLASSType(c->dtype()) << "*)&(";
        (*this)(c);
        os() << "), ";
        (*this)(lda);
        os() << ", ";
        (*this)(ldb);
        os() << ", ";
        (*this)(stridea);
        os() << ", ";
        (*this)(strideb);
        os() << ", ";
        (*this)(stridec);
        os() << ", ";
        (*this)(op->alpha_);
        os() << ", ";
        (*this)(op->beta_);
        os() << ", ";
        (*this)(prop->warpIdBatch_);
        os() << ", ";
        (*this)(prop->warpIdM_);
        os() << ", ";
        (*this)(prop->warpIdN_);
        os() << ", ";
        (*this)(prop->laneId_);
        os() << ");" << std::endl;

        neededMicroKernels_.emplace_back("matmul/cutlass/gemm.h");
        break;
    }

    case MatMulBackend::CutlassMicroBlock:
        ERROR("CutlassMicroBlock should be lowered before codegen");

    default:
        inMatmul_ = false;
        throw InvalidProgram("MatMul backend " +
                             freetensor::toString(op->backend_) +
                             " is not supported for GPU");
    }

    inMatmul_ = false;
}

NativeCode codeGenCUDA(const Func &func, const Ref<Target> &_target) {
    ASSERT(_target->type() == TargetType::GPU);
    auto target = _target.as<GPUTarget>();

    auto prefix = mangle(func->name_);
    auto nParams = func->params_.size();

    CodeGenCUDA visitor(func->params_, func->returns_, target, prefix);
    auto &&op = func->body_;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    std::string header = R"~~~(
#include <gpu_runtime.h>

extern __shared__ uint8_t __shmem[];

extern "C" {
)~~~";
    std::string tailer = R"~~~(
}
)~~~";

    for (auto &&item : visitor.neededMicroKernels()) {
        header = "#include \"micro_kernel/" + item + "\"\n" + header;
    }

    auto body = visitor.toString([&](const CodeGenCUDA::Stream &stream) {
        if (stream.name_ == "default") {
            std::string s =
                "void " + prefix +
                "_run(void **__params, void **returns, size_t **retShapes, "
                "size_t *retDims, GPUContext_t ctx) {\n";
            // We copy `__params` to `params`, in order to pass the parameter
            // pack into a kernel
            s += "__ByValArray<void *, " + std::to_string(nParams) +
                 "> params;\n";
            for (size_t i = 0; i < nParams; i++) {
                s += "params[" + std::to_string(i) + "] = __params[" +
                     std::to_string(i) + "];\n";
            }
            s += "\n";

            // Set the default stream to __stream, in order to implement
            // multiple streams
            s += "cudaStream_t __stream = 0;\n";
            s += "\n";

            s += "uint8_t *__glmem = (uint8_t*)ctx->gpuGlobalStaticPool();\n";

            s += stream.os_.str();
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
            for (auto &&[name, d] : stream.useDefs_) {
                os << (first ? "" : ", ");
                auto &&buffer = d->buffer_;
                auto &&tensor = buffer->tensor();
                auto &&shape = tensor->shape();

                switch (buffer->mtype()) {
                case MemType::ByValue:
                    // e.g.
                    // __ByValArray<__ByValArray<float, 2>, 2> x;
                    for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                        os << "__ByValArray<";
                    }
                    os << visitor.gen(tensor->dtype());
                    for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                        ASSERT((*it)->nodeType() == ASTNodeType::IntConst);
                        os << ", " << (*it).as<IntConstNode>()->val_ << ">";
                    }
                    os << " " << mangle(name);
                    break;

                default:
                    // e.g. mdspan<float, extents<5, 5>> x
                    os << visitor.genMdPtrType(d, !isWritable(buffer->atype()))
                       << " " << mangle(name);
                }
                first = false;
            }
            for (auto &&name : stream.useIters_) {
                os << (first ? "" : ", ") << "int " << mangle(name);
                first = false;
            }
            os << ", __ByValArray<void *, " + std::to_string(nParams) +
                      "> params, uint8_t *__glmem) ";
            os << stream.os_.str() << std::endl;
            return os.str();
        }
    });

    // Pre-allocate static gpu/global memory pool
    // TODO: Support dynamic size and allocate the dynamic part at run time
    auto globalSize = visitor.globalSize();
    ASSERT(globalSize->nodeType() == ASTNodeType::IntConst);
    StaticInfo staticInfo;
    staticInfo.gpuGlobalStaticPoolSize_ = globalSize.as<IntConstNode>()->val_;

    return NativeCode::fromFunc(func, header + body + tailer, prefix + "_run",
                                target, staticInfo);
}

} // namespace freetensor

#endif // FT_WITH_CUDA
