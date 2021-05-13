#include <codegen/code_gen_cpu.h>
#include <pass/simplify.h>

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

std::pair<std::string, std::vector<std::string>> codeGenCPU(const Stmt &op) {
    CodeGenCPU visitor;
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

    auto body = visitor.toString([&](const CodeGenCPU::Stream &stream) {
        return "void run(void **_params) " + stream.os_.str();
    });
    return std::make_pair(header + body + tailer, visitor.params());
}

} // namespace ir
