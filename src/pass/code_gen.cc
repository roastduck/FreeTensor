#include <pass/code_gen.h>

namespace ir {

void CodeGen::beginBlock() {
    os << "{" << std::endl;
    nIndent++;
}

void CodeGen::endBlock() {
    nIndent--;
    makeIndent();
    os << "}" << std::endl;
}

void CodeGen::makeIndent() {
    for (int i = 0; i < nIndent; i++) {
        os << "  ";
    }
}

std::string CodeGen::toString() { return os.str(); }

} // namespace ir

