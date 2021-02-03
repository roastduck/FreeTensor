#include <codegen/code_gen_cpu.h>

namespace ir {

void CodeGenCPU::visit(const For &op) {
    if (op->parallel_ == "openmp") {
        os() << "#pragma omp parallel for" << std::endl;
    }
    CodeGenC::visit(op);
}

std::pair<std::string, std::vector<std::string>> codeGenCPU(const AST &op) {
    CodeGenCPU visitor;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    // TODO: Pure C?
    const char *header = "#include <cstdint>\n"
                         "#include <algorithm>\n" // min, max
                         "#include <array>\n"     // ByValue
                         "#define restrict __restrict__\n"
                         "#define __ByValArray std::array\n"
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

