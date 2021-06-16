#include <codegen/code_gen_cpu.h>
#include <pass/simplify.h>

#include "detail/code_gen_c.h"

namespace ir {

void CodeGenCPU::visit(const ReduceTo &op) {
    if (op->atomic_) {
        os() << "#pragma omp atomic" << std::endl;
    }
    CodeGenC::visit(op);
}

void CodeGenCPU::visit(const For &op) {
    if (op->parallel_ == "openmp") {
        os() << "#pragma omp parallel for" << std::endl;
    } else if (op->vectorize_) {
        os() << "#pragma omp simd" << std::endl;
    } else if (op->unroll_) {
        os() << "#pragma GCC unroll " << op->len_ << std::endl;
    }
    CodeGenC::visit(op);
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
