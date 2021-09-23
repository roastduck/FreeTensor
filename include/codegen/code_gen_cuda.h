#ifndef CODE_GEN_CUDA_H
#define CODE_GEN_CUDA_H

#include <unordered_map>

#include <codegen/code_gen_c.h>
#include <func.h>

namespace ir {

struct CodeGenCUDAStream : public CodeGenStream {
    std::unordered_map<std::string, int> threadDim_;
    int sharedSize_ = 0;
};

class CodeGenCUDA : public CodeGenC<CodeGenCUDAStream> {
  public:
    typedef CodeGenCUDAStream Stream;

  private:
    int nKernel_ = 0;
    int sharedStackTop_ = 0;

  public:
    CodeGenCUDA(const std::vector<std::string> &params) : CodeGenC(params) {}

  private:
    bool inKernel() const;

  protected:
    using CodeGenC<CodeGenCUDAStream>::visit;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
    void visit(const MatMul &op) override;
};

/**
 * Generate target function code
 *
 * @return : source
 */
std::string codeGenCUDA(const Func &func);

} // namespace ir

#endif // CODE_GEN_CUDA_H
