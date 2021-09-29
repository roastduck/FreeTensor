#ifndef GPU_LOWER_VECTOR_H
#define GPU_LOWER_VECTOR_H

#include <unordered_map>

#include <analyze/analyze_linear.h>
#include <func.h>
#include <pass/z3_simplify.h>

namespace ir {

namespace gpu {

class LowerVector : public Z3Simplify {
    static constexpr int VEC_LEN[] = {4, 2};

    std::string var_;
    Expr begin_;
    uint64_t varHash_;
    int vecLen_, isIndex_ = 0;
    bool simplifyOnly_ = false;

    std::unordered_map<std::string, Ref<Buffer>> buffers_;

    AnalyzeLinear analyzeLinear_;

  private:
    std::string vecType(DataType dtype) const;
    bool hasVectorIndex(const Expr &index);
    Expr getIndex(const Expr &index);

  protected:
    using Z3Simplify::visit;

    Stmt visit(const For &op) override;
    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt lowerVector(const Stmt &op);

DEFINE_PASS_FOR_FUNC(lowerVector)

} // namespace gpu

} // namespace ir

#endif // GPU_LOWER_VECTOR_H
