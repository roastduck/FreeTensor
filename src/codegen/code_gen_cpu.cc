#include <codegen/code_gen_cpu.h>
#include <pass/simplify.h>

#include "detail/code_gen_c.h"

namespace ir {

#ifdef WITH_MKL

static char genMKLTypeMark(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return 's';
    default:
        ASSERT(false);
    }
}

#endif

void CodeGenCPU::visit(const ReduceTo &op) {
    if (op->atomic_) {
        os() << "#pragma omp atomic" << std::endl;
    }
    CodeGenC::visit(op);
}

void CodeGenCPU::visit(const For &op) {
    if (op->parallel_ == "openmp") {
        os() << "#pragma omp parallel for" << std::endl;
        bool oldInParallel = inParallel_;
        inParallel_ = true;
        CodeGenC::visit(op);
        inParallel_ = oldInParallel;
        return;
    } else if (op->vectorize_) {
        os() << "#pragma omp simd" << std::endl;
    } else if (op->unroll_) {
        os() << "#pragma GCC unroll " << op->len_ << std::endl;
    }
    CodeGenC::visit(op);
}

void CodeGenCPU::visit(const MatMul &op) {
#ifdef WITH_MKL
    makeIndent();
    if (inParallel_) {
        os() << "mkl_set_num_threads_local(1);" << std::endl;
        // TODO: set it to max(1, cpu_count / outer_threads_count)
    } else {
        os() << "mkl_set_num_threads_local(0); // 0 == reset" << std::endl;
    }

    auto d = dtype(op->c_);
    if (dtype(op->a_) != d || dtype(op->b_) != d) {
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
    CodeGenCPU visitor(func->params_);
    auto &&op = func->body_;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    // TODO: Pure C?
    const char *header = R"~~~(
#include <cpu_runtime.h>

extern "C" {
)~~~";
    const char *tailer = R"~~~(
}
)~~~";

    auto body = visitor.toString([&](const CodeGenStream &stream) {
        return "void run(void **_params) " + stream.os_.str();
    });
    return header + body + tailer;
}

} // namespace ir
