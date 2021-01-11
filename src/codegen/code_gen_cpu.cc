#include <codegen/code_gen_cpu.h>

namespace ir {

void CodeGenCPU::visit(const For &op) {
    if (op->parallel_ == "openmp") {
        os << "#pragma omp parallel for" << std::endl;
    }
    CodeGenC::visit(op);
}

std::pair<std::string, std::vector<std::string>> codeGenCPU(const AST &op) {
    CodeGenCPU visitor;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = "#include <cstdint>\n"
                         "#include <algorithm>\n" // TODO: Pure C?
                         "#define restrict __restrict__\n"
                         "\n"
                         "extern \"C\" {\n"
                         "\n";
    const char *tailer = "\n"
                         "}";

    return std::make_pair((std::string)header + "void run(void **_params) " +
                              visitor.toString() + tailer,
                          visitor.params());
}

} // namespace ir

