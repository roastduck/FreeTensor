#ifndef CODE_GEN_CPU_H
#define CODE_GEN_CPU_H

#include <unordered_set>

#include <codegen/code_gen_c.h>
#include <func.h>

namespace ir {

class CodeGenCPU : public CodeGenC<CodeGenStream> {
    bool inParallel_ = false;
    int64_t sharedStackTop_ = 8192 * 1024, sharedStackSize_ = 0;
    int64_t threadStackTop_ = 0, threadStackSize_ = 0;
    std::unordered_set<ID> collapsed_;

  public:
    CodeGenCPU(const std::vector<std::string> &params,
               const std::vector<std::pair<std::string, DataType>> &returns)
        : CodeGenC(params, returns) {}

    int64_t sharedStackSize() const { return sharedStackSize_; }
    int64_t threadStackSize() const { return threadStackSize_; }

  protected:
    void genAlloc(const Tensor &tensor, const std::string &rawPtr,
                  const std::string &shapePtr,
                  const std::string &dimPtr) override;

    using CodeGenC<CodeGenStream>::visit;
    void visit(const VarDef &op) override;
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
