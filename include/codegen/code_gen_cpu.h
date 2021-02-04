#ifndef CODE_GEN_CPU_H
#define CODE_GEN_CPU_H

#include <codegen/code_gen_c.h>

namespace ir {

class CodeGenCPU : public CodeGenC {
  protected:
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
};

/**
 * Generate target function code
 *
 * @return : (source, list of params)
 */
std::pair<std::string, std::vector<std::string>> codeGenCPU(const AST &op);

} // namespace ir

#endif // CODE_GEN_CPU_H
