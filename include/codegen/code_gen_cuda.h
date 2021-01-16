#ifndef CODE_GEN_CUDA_H
#define CODE_GEN_CUDA_H

#include <unordered_map>

#include <codegen/code_gen_c.h>

namespace ir {

class CodeGenCUDA : public CodeGenC {
    struct ThreadInfo {
        std::string threadId_;
        Expr offset_;
    };

    std::unordered_map<std::string, ThreadInfo> varMap_;
    int nKernel_ = 0;

  private:
    bool inKernel() const;

  protected:
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const Var &op) override;
    void visit(const For &op) override;
};

/**
 * Generate target function code
 *
 * @return : (source, list of params)
 */
std::pair<std::string, std::vector<std::string>> codeGenCUDA(const AST &op);

} // namespace ir

#endif // CODE_GEN_CUDA_H
