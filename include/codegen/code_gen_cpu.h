#ifndef CODE_GEN_CPU_H
#define CODE_GEN_CPU_H

#include <codegen/code_gen_c.h>
#include <func.h>

namespace ir {

class CodeGenCPU : public CodeGenC<CodeGenStream> {
    bool inParallel_ = false;

  public:
    CodeGenCPU(const std::vector<std::string> &params) : CodeGenC(params) {}

  protected:
    using CodeGenC<CodeGenStream>::visit;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
    void visit(const MatMul &op) override;
};

/**
 * Generate target function code
 *
 * @return : source
 */
std::string codeGenCPU(const Func &func);

} // namespace ir

#endif // CODE_GEN_CPU_H
