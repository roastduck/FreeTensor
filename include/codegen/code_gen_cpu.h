#ifndef FREE_TENSOR_CODE_GEN_CPU_H
#define FREE_TENSOR_CODE_GEN_CPU_H

#include <unordered_set>

#include <codegen/code_gen_c.h>
#include <func.h>

namespace freetensor {

class CodeGenCPU : public CodeGenC<CodeGenStream> {
    bool inParallel_ = false;
    int64_t sharedStackTop_ = 8192 * 1024, sharedStackSize_ = 0;
    int64_t threadStackTop_ = 0, threadStackSize_ = 0;
    std::unordered_set<For> collapsed_;
    std::unordered_set<VarDef> usedAsReduction_;

  public:
    CodeGenCPU(const std::vector<FuncParam> &params,
               const std::vector<FuncRet> &returns)
        : CodeGenC(params, returns) {}

    int64_t sharedStackSize() const { return sharedStackSize_; }
    int64_t threadStackSize() const { return threadStackSize_; }

  protected:
    void genAlloc(const Ref<Tensor> &tensor, const std::string &rawPtr,
                  const std::string &shapePtr,
                  const std::string &dimPtr) override;

    void genScalar(const VarDef &def,
                   const std::vector<Expr> &indices) override;

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

} // namespace freetensor

#endif // FREE_TENSOR_CODE_GEN_CPU_H
