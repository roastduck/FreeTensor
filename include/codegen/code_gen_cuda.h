#ifndef FREE_TENSOR_CODE_GEN_CUDA_H
#define FREE_TENSOR_CODE_GEN_CUDA_H

#include <unordered_map>
#include <unordered_set>

#include <codegen/code_gen_c.h>
#include <func.h>

namespace freetensor {

struct CodeGenCUDAStream : public CodeGenStream {
    std::unordered_map<ParallelScope, Expr> threadDim_;
    int64_t sharedSize_ = 0;
};

class CodeGenCUDA : public CodeGenC<CodeGenCUDAStream> {
  public:
    typedef CodeGenCUDAStream Stream;

  private:
    int nKernel_ = 0;
    int64_t sharedStackTop_ = 0, globalStackTop_ = 0;
    int64_t globalSize_ = 0;
    std::unordered_set<Stmt> streamScopes_;
    bool inCublas_ = false;

  public:
    CodeGenCUDA(const std::vector<FuncParam> &params,
                const std::vector<FuncRet> &returns)
        : CodeGenC(params, returns) {}

    using CodeGenC<CodeGenCUDAStream>::genMdPtrType;

    int64_t globalSize() const { return globalSize_; }

  private:
    bool inKernel() const;

    void exprOr1(const std::unordered_map<ParallelScope, Expr> &dict,
                 const ParallelScope &key);

  protected:
    void genAlloc(const Ref<Tensor> &tensor, const std::string &rawPtr,
                  const std::string &shapePtr,
                  const std::string &dimPtr) override;

    using CodeGenC::genScalar;
    void genScalar(const VarDef &def,
                   const std::vector<Expr> &indices) override;

    using CodeGenC<CodeGenCUDAStream>::visit;
    void visitStmt(const Stmt &stmt) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
    void visit(const MatMul &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const Alloc &op) override;
    void visit(const Free &op) override;
};

/**
 * Generate target function code
 *
 * @return : source
 */
std::string codeGenCUDA(const Func &func);

} // namespace freetensor

#endif // FREE_TENSOR_CODE_GEN_CUDA_H
