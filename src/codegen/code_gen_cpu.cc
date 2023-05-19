#include <analyze/all_uses.h>
#include <codegen/code_gen_cpu.h>
#include <container_utils.h>
#include <math/utils.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

#include "detail/code_gen_c.h"

namespace freetensor {

#ifdef FT_WITH_MKL

static char genMKLTypeMark(DataType dtype) {
    switch (dtype.base()) {
    case DataType::Float64:
        return 'd';
    case DataType::Float32:
        return 's';
    default:
        ASSERT(false);
    }
}

#endif

void CodeGenCPU::genAlloc(const Ref<Tensor> &tensor, const std::string &rawPtr,
                          const std::string &shapePtr,
                          const std::string &dimPtr) {
    // Allocate for return values. The allocated buffer is moved as `Array`s,
    // which will be free'd using `delete[]`, so we are using `new` here.
    auto ndim = tensor->shape().size();
    makeIndent();
    os() << dimPtr << " = " << ndim << ";" << std::endl;
    makeIndent();
    os() << shapePtr << " = " << ndim << " > 0 ? (new size_t[" << ndim
         << "]) : NULL;" << std::endl;
    makeIndent();
    os() << rawPtr << " = new std::byte[";
    for (auto &&[i, dim] : views::enumerate(tensor->shape())) {
        os() << "(" << shapePtr << "[" << i << "] = ";
        (*this)(dim);
        os() << ") * ";
    }
    os() << "sizeof(" << gen(tensor->dtype()) << ")];" << std::endl;
}

void CodeGenCPU::genScalar(const VarDef &def,
                           const std::vector<Expr> &indices) {
    if (usedAsReduction_.count(def)) {
        this->os() << mangle(def->name_) + "_arrptr";
        for (auto &&index : indices) {
            this->os() << "[";
            (*this)(index);
            this->os() << "]";
        }
    } else {
        BaseClass::genScalar(def, indices);
    }
}

void CodeGenCPU::visit(const VarDef &op) {
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor->shape();

    if (isInputting(op->buffer_->atype()) ||
        isOutputting(op->buffer_->atype()) || op->viewOf_.has_value() ||
        shape.empty()) {
        BaseClass::visit(op);

    } else {
        auto name = mangle(op->name_);

        switch (op->buffer_->mtype()) {
        case MemType::CPUHeap:
            // e.g. UncheckedOpt<mdspan_r<float, std::extents<5, 5>>> x_opt;
            //      auto &x = *x_opt;
            this->makeIndent();
            this->os() << "UncheckedOpt<" << genMdPtrType(op) << "> " << name
                       << "_opt;" << std::endl;
            this->makeIndent();
            this->os() << "auto &" << name << " = *" << name << "_opt;"
                       << std::endl;
            this->markDef(op);
            (*this)(op->body_);
            this->markUndef(op);
            break;

        case MemType::CPU: {
            // e.g.
            // auto &&x = mdspan_r<float, std::extents<5, 5,
            // 5>>(&__threadStack[0]);
            this->makeIndent();
            this->os() << "auto &&" << name << " = ";
            std::string rawPtr;
            if (inParallel_) {
                rawPtr = "&__threadStack[omp_get_thread_num()][" +
                         std::to_string(threadStackTop_) + "]";
            } else {
                rawPtr =
                    "&__sharedStack[" + std::to_string(sharedStackTop_) + "]";
            }
            this->genMdPtrDef(op, rawPtr);
            this->os() << ";" << std::endl;

            int64_t size = sizeOf(tensor->dtype());
            for (auto &&dim : shape) {
                if (dim->nodeType() == ASTNodeType::IntConst) {
                    size *= dim.as<IntConstNode>()->val_;
                } else {
                    ERROR("BUG: Dyanmic sized variables cannot be allocated on "
                          "stack. Should be transformed to heap-allocated in "
                          "pass/make_heap_alloc");
                }
            }

            // Align to 64 bytes (TODO: look up cache line size from Target)
            size = ceilDiv<int64_t>(size, 64) * 64;

            if (inParallel_) {
                threadStackSize_ =
                    std::max(threadStackSize_, threadStackTop_ + size);
                threadStackTop_ += size;
                this->markDef(op);
                (*this)(op->body_);
                this->markUndef(op);
                threadStackTop_ -= size;
            } else {
                sharedStackSize_ =
                    std::max(sharedStackSize_, sharedStackTop_ + size);
                sharedStackTop_ += size;
                this->markDef(op);
                (*this)(op->body_);
                this->markUndef(op);
                sharedStackTop_ -= size;
            }
            break;
        }

        default:
            BaseClass::visit(op);
            break;
        }
    }
}

void CodeGenCPU::visit(const ReduceTo &op) {
    if (op->sync_) {
        switch (op->op_) {
        case ReduceOp::Add:
        case ReduceOp::Mul:
        case ReduceOp::LAnd:
        case ReduceOp::LOr:
            // Supported by `omp atomic`
            os() << "#pragma omp atomic" << std::endl;
            BaseClass::visit(op);
            break;

        // The followings are not supported by `omp atomic`, do atomic CAS by
        // ourselves. `atomicUpdate` is defined in `runtime/cpu_runtime.h`
        case ReduceOp::Min:
            makeIndent();
            os() << "atomicUpdate(";
            genScalar(op);
            // User names are prefixed by an `_`, so we are safe with `x` here
            os() << ", [&](auto &&x) { return std::min(x, ";
            (*this)(op->expr_);
            os() << "); });" << std::endl;
            break;
        case ReduceOp::Max:
            makeIndent();
            os() << "atomicUpdate(";
            genScalar(op);
            // User names are prefixed by an `_`, so we are safe with `x` here
            os() << ", [&](auto &&x) { return std::max(x, ";
            (*this)(op->expr_);
            os() << "); });" << std::endl;
            break;

        default:
            ASSERT(false);
        }
    } else {
        BaseClass::visit(op);
    }
}

void CodeGenCPU::visit(const For &op) {
    if (std::holds_alternative<OpenMPScope>(op->property_->parallel_) &&
        !collapsed_.count(op)) {
        Expr totLen = op->len_;

        int collapse = 1;
        for (Stmt inner = op->body_;
             inner->nodeType() == ASTNodeType::For &&
             std::holds_alternative<OpenMPScope>(
                 inner.as<ForNode>()->property_->parallel_);
             inner = inner.as<ForNode>()->body_) {
            Expr innerLen = inner.as<ForNode>()->len_;
            if (allIters(innerLen).count(op->iter_)) {
                // Collapsed inner loops' length should not depend on the outer
                // loop
                break;
            }
            totLen = makeMul(totLen, innerLen);
            collapse++;
            collapsed_.insert(inner.as<ForNode>());
            if (!inner.as<ForNode>()->property_->reductions_.empty())
                ERROR("Collapsed inner parallel loop should not have reduction "
                      "items.");
        }

        for (auto &&r : op->property_->reductions_) {
            if (!buffer(r->var_)->tensor()->shape().empty()) {
                usedAsReduction_.insert(def(r->var_));
                auto var = mangle(r->var_);
                makeIndent();
                os() << "auto &&" << var << "_arrptr = toArrPtr(" << var << ");"
                     << std::endl;
            }
        }
        os() << "#pragma omp parallel for";
        if (collapse > 1) {
            os() << " collapse(" << collapse << ")";
        }
        // As per OpenMP specification, even the loop's length is only 2, it
        // still synchronizes all the threads in the implicit barrier. Thus we
        // need to explicitly set `num_threads` here. It will not affect the
        // semantics as long as we don't use `no_wait`.
        os() << " num_threads(";
        (*this)(makeMin(totLen, makeIntrinsic("omp_get_max_threads()", {},
                                              DataType::Int32, false)));
        os() << ")";
        if (!op->property_->reductions_.empty()) {
            for (size_t i = 1, n = op->property_->reductions_.size(); i < n;
                 i++) {
                if (op->property_->reductions_[i]->op_ !=
                    op->property_->reductions_.front()->op_) {
                    throw InvalidProgram(
                        "Reduction operators of each parallel reduction "
                        "variables should be the same in a single OpenMP loop");
                }
            }
            os() << " reduction(";
            switch (op->property_->reductions_.front()->op_) {
            case ReduceOp::Add:
                os() << "+: ";
                break;
            case ReduceOp::Mul:
                os() << "*: ";
                break;
            case ReduceOp::Min:
                os() << "min: ";
                break;
            case ReduceOp::Max:
                os() << "max: ";
                break;
            case ReduceOp::LAnd:
                os() << "&&: ";
                break;
            case ReduceOp::LOr:
                os() << "||: ";
                break;
            default:
                ASSERT(false);
            }
            bool first = true;
            for (auto &&r : op->property_->reductions_) {
                if (!first) {
                    os() << ", ";
                }
                first = false;
                if (!buffer(r->var_)->tensor()->shape().empty()) {
                    os() << mangle(r->var_) << "_arrptr";
                    for (auto &&[b, e] : views::zip(r->begins_, r->ends_)) {
                        os() << "[";
                        (*this)(b);
                        os() << ":";
                        // Note that OpenMP accepts `[begin : length]` rather
                        // than
                        // `[begin : end]`
                        (*this)(makeSub(e, b));
                        os() << "]";
                    }
                } else {
                    os() << mangle(r->var_);
                }
            }
            os() << ")";
        }
        os() << std::endl;
        bool oldInParallel = inParallel_;
        inParallel_ = true;
        BaseClass::visit(op);
        inParallel_ = oldInParallel;
        for (auto &&r : op->property_->reductions_) {
            if (!buffer(r->var_)->tensor()->shape().empty()) {
                usedAsReduction_.erase(def(r->var_));
            }
        }
        return;
    } else if (op->property_->vectorize_) {
        os() << "#pragma omp simd" << std::endl;
    } else if (op->property_->unroll_) {
        os() << "#pragma GCC unroll " << op->len_ << std::endl;
    }
    BaseClass::visit(op);
}

void CodeGenCPU::visit(const MatMul &op) {
#ifdef FT_WITH_MKL
    makeIndent();
    if (inParallel_) {
        os() << "mkl_set_num_threads_local(1);" << std::endl;
        // TODO: set it to max(1, cpu_count / outer_threads_count)
    } else {
        os() << "mkl_set_num_threads_local(0); // 0 == reset" << std::endl;
    }

    auto d = op->c_->dtype();
    if (op->a_->dtype() != d || op->b_->dtype() != d) {
        throw InvalidProgram(
            "MKL requires all matrices have the same data type");
    }

    bool transA = !op->aIsRowMajor_, transB = !op->bIsRowMajor_;
    Expr a = op->a_, b = op->b_, c = op->c_;
    Expr m = op->m_, k = op->k_, n = op->n_;
    Expr lda = op->lda_, ldb = op->ldb_, ldc = op->ldc_;
    Expr stridea = op->stridea_, strideb = op->strideb_, stridec = op->stridec_;
    if (!op->cIsRowMajor_) {
        transA = !transA;
        transB = !transB;
        std::swap(transA, transB);
        std::swap(a, b);
        std::swap(lda, ldb);
        std::swap(stridea, strideb);
        std::swap(n, m);
    }

    makeIndent();
    os() << "cblas_" << genMKLTypeMark(d)
         << "gemm_batch_strided(CblasRowMajor, "
         << (transA ? "CblasTrans" : "CblasNoTrans") << ", "
         << (transB ? "CblasTrans" : "CblasNoTrans") << ", ";
    (*this)(m);
    os() << ", ";
    (*this)(n);
    os() << ", ";
    (*this)(k);
    os() << ", ";
    (*this)(op->alpha_);
    os() << ", &";
    (*this)(a);
    os() << ", ";
    (*this)(lda);
    os() << ", ";
    (*this)(stridea);
    os() << ", &";
    (*this)(b);
    os() << ", ";
    (*this)(ldb);
    os() << ", ";
    (*this)(strideb);
    os() << ", ";
    (*this)(op->beta_);
    os() << ", &";
    (*this)(c);
    os() << ", ";
    (*this)(ldc);
    os() << ", ";
    (*this)(stridec);
    os() << ", ";
    (*this)(op->batchSize_);
    os() << ");" << std::endl;
#else
    ERROR("Configuring with MKL is needed");
#endif
}

std::string codeGenCPU(const Func &func) {
    CodeGenCPU visitor(func->params_, func->returns_);
    auto &&op = func->body_;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = R"~~~(
#include <cpu_runtime.h>

extern "C" {
)~~~";
    const char *tailer = R"~~~(
}
)~~~";

    auto body = visitor.toString([&](const CodeGenStream &stream) {
        std::string s;
        if (visitor.sharedStackSize() > 0) {
            s += "static uint8_t *__sharedStack = nullptr;\n";
        }
        if (visitor.threadStackSize() > 0) {
            s += "static uint8_t **__threadStack = nullptr;\n";
        }
        s += "__attribute__((constructor)) static void initStack() {\n";
        if (visitor.sharedStackSize() > 0) {
            s += "  __sharedStack = new uint8_t[" +
                 std::to_string(visitor.sharedStackSize()) + "];\n";
        }
        if (visitor.threadStackSize() > 0) {
            s += "  __threadStack = new uint8_t *[omp_get_max_threads()];\n";
            s += "  #pragma omp parallel\n";
            s += "  __threadStack[omp_get_thread_num()] = new uint8_t[" +
                 std::to_string(visitor.threadStackSize()) + "];\n";
        }
        s += "}\n";
        s += "__attribute__((destructor)) static void deinitStack() {\n";
        if (visitor.sharedStackSize() > 0) {
            s += "  delete[] __sharedStack;\n";
        }
        if (visitor.threadStackSize() > 0) {
            s += "  #pragma omp parallel\n";
            s += "  delete[] __threadStack[omp_get_thread_num()];\n";
            s += "  delete[] __threadStack;\n";
        }
        s += "}\n";
        s += "void run(void **params, void **returns, size_t **retShapes, "
             "size_t *retDims, CPUContext_t ctx) {\n";
        s += stream.os_.str();
        s += "}";
        return s;
    });
    return header + body + tailer;
}

} // namespace freetensor
