#include <codegen/code_gen.h>

namespace ir {

void CodeGen::beginBlock() {
    os() << "{" << std::endl;
    nIndent_++;
}

void CodeGen::endBlock() {
    nIndent_--;
    makeIndent();
    os() << "}" << std::endl;
}

void CodeGen::makeIndent() {
    for (int i = 0; i < nIndent_; i++) {
        os() << "  ";
    }
}

std::string CodeGen::toString() { return os().str(); }

} // namespace ir

