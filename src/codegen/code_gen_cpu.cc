#include <codegen/code_gen_cpu.h>
#include <container_utils.h>
#include <math/utils.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

#include "detail/code_gen_c.h"

namespace freetensor {

#ifdef FT_WITH_MKL

static char genMKLTypeMark(DataType dtype) {
    switch (dtype) {
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
    auto ndim = tensor->shape().size();
    makeIndent();
    os() << shapePtr << " = " << ndim << " > 0 ? (size_t*)malloc((" << dimPtr
         << " = " << ndim << ") * sizeof(size_t)) : NULL;" << std::endl;
    makeIndent();
    os() << rawPtr << " = malloc(";
    for (auto &&[i, dim] : views::enumerate(tensor->shape())) {
        os() << "(" << shapePtr << "[" << i << "] = ";
        (*this)(dim);
        os() << ") * ";
    }
    os() << "sizeof(" << gen(tensor->dtype()) << "));" << std::endl;
}

void CodeGenCPU::genScalar(const VarDef &def,
                           const std::vector<Expr> &indices) {
    if (usedAsReduction_.count(def)) {
        this->os() << mangle(def->name_) + "_ptr";
        for (auto &&index : indices) {
            this->os() << "[";
            (*this)(index);
            this->os() << "]";
        }
    } else {
        CodeGenC<CodeGenStream>::genScalar(def, indices);
    }
}

void CodeGenCPU::visit(const VarDef &op) {
    auto &&tensor = op->buffer_->tensor();
    auto &&shape = tensor->shape();

    if (op->buffer_->atype() != AccessType::Cache || shape.empty()) {
        CodeGenC::visit(op);

    } else {
        auto name = mangle(op->name_);

        switch (op->buffer_->mtype()) {
        case MemType::CPUHeap:
            // e.g. UncheckedOpt<mdspan_r<float, std::extents<5, 5>>> x_opt;
            //      auto &x = *x_opt;
            this->makeIndent();
            this->os() << "UncheckedOpt<";
            genMdPtrType(op->buffer_);
            this->os() << "> " << name << "_opt;" << std::endl;
            this->makeIndent();
            this->os() << "auto &" << name << " = *" << name << "_opt;"
                       << std::endl;
            this->markDefBuffer(op);
            (*this)(op->body_);
            this->markUndefBuffer(op);
            break;

        case MemType::CPU: {
            // e.g.
            // auto x = mdspan_r<float, std::extents<5, 5, 5>>(&__stack[200 +
            // omp_get_thread_num() * _threadStackTop + 100]);
            this->makeIndent();
            this->os() << "auto " << name << " = ";
            std::string rawPtr;
            if (inParallel_) {
                rawPtr = "&__stack[" + std::to_string(sharedStackTop_) +
                         " + omp_get_thread_num() * _threadStackSize + " +
                         std::to_string(threadStackTop_) + "]";
            } else {
                rawPtr = "&__stack[" + std::to_string(sharedStackTop_) + "]";
            }
            this->genMdPtrDef(op->buffer_, rawPtr);
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
                this->markDefBuffer(op);
                (*this)(op->body_);
                this->markUndefBuffer(op);
                threadStackTop_ -= size;
            } else {
                sharedStackSize_ =
                    std::max(sharedStackSize_, sharedStackTop_ + size);
                sharedStackTop_ += size;
                this->markDefBuffer(op);
                (*this)(op->body_);
                this->markUndefBuffer(op);
                sharedStackTop_ -= size;
            }
            break;
        }

        default:
            CodeGenC::visit(op);
            break;
        }
    }
}

void CodeGenCPU::visit(const ReduceTo &op) {
    if (op->atomic_) {
        os() << "#pragma omp atomic" << std::endl;
        // FIXME: OpenMP supports atomic min and max only for FORTRAN
    }
    CodeGenC::visit(op);
}

void CodeGenCPU::visit(const For &op) {
    if (std::holds_alternative<OpenMPScope>(op->property_->parallel_) &&
        !collapsed_.count(op)) {
        int collapse = 1;
        for (Stmt inner = op->body_;
             inner->nodeType() == ASTNodeType::For &&
             std::holds_alternative<OpenMPScope>(
                 inner.as<ForNode>()->property_->parallel_);
             inner = inner.as<ForNode>()->body_) {
            collapse++;
            collapsed_.insert(inner.as<ForNode>());
        }

        for (auto &&r : op->property_->reductions_) {
            if (!buffer(r->var_)->tensor()->shape().empty()) {
                usedAsReduction_.insert(def(r->var_));
                auto var = mangle(r->var_);
                makeIndent();
                os() << "auto " << var << "_ptr = toArrPtr(" << var << ");"
                     << std::endl;
            }
        }
        os() << "#pragma omp parallel for";
        if (collapse > 1) {
            os() << " collapse(" << collapse << ")";
        }
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
            case ReduceOp::Sub:
                os() << "-: ";
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
                    os() << mangle(r->var_) << "_ptr";
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
        CodeGenC::visit(op);
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
    CodeGenC::visit(op);
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
        std::string s =
            "void run(void **_params, void **_returns, size_t **_retShapes, "
            "size_t *_retDims, CPUContext_t _ctx) {\n";
        s += "  size_t _sharedStackSize = " +
             std::to_string(visitor.sharedStackSize()) + ";\n";
        s += "  size_t _threadStackSize = " +
             std::to_string(visitor.threadStackSize()) + ";\n";
        s += "  auto __stack = new uint8_t[_sharedStackSize + "
             "omp_get_max_threads() * _threadStackSize];\n";
        s += stream.os_.str();
        s += "  delete[] __stack;\n";
        s += "}";
        return s;
    });
    return header + body + tailer;
}

} // namespace freetensor
