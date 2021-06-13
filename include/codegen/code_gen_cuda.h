#ifndef CODE_GEN_CUDA_H
#define CODE_GEN_CUDA_H

#include <unordered_map>

#include <codegen/code_gen_c.h>
#include <func.h>

namespace ir {

class CodeGenCUDA : public CodeGenC {
    int nKernel_ = 0;

  public:
    CodeGenCUDA(const std::vector<std::string> &params) : CodeGenC(params) {}

  private:
    bool inKernel() const;

  protected:
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const Sqrt &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
};

/**
 * Generate target function code
 *
 * @return : source
 */
std::string codeGenCUDA(const Func &func);

} // namespace ir

#endif // CODE_GEN_CUDA_H
