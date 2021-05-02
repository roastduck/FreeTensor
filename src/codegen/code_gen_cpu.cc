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
    }
    if (op->unroll_) {
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
    const char *header =
        "#include <cstdint>\n"
        "#include <cassert>\n"
        "#include <algorithm>\n" // min, max
        "#include <array>\n"     // ByValue
        "#define restrict __restrict__\n"
        "#define __ByValArray std::array\n"
        "\n"
        "template <class T>\n"
        "T floorDiv(T a, T b) {\n"
        "  T res = a / b, rem = a % b;\n"
        "  return res - (rem != 0 && ((rem < 0) != (b < 0)));\n"
        "}\n"
        "template <class T>\n"
        "T ceilDiv(T a, T b) {\n"
        "  T res = a / b, rem = a % b;\n"
        "  return res + (rem != 0 && ((rem < 0) == (b < 0)));\n"
        "}\n"
        "\n"
        "extern \"C\" {\n"
        "\n";
    const char *tailer = "\n"
                         "}";

    auto body = visitor.toString([&](const CodeGenCPU::Stream &stream) {
        return "void run(void **_params) " + stream.os_.str();
    });
    return std::make_pair(header + body + tailer, visitor.params());
}

} // namespace ir
