#ifndef FREE_TENSOR_GPU_NORMALIZE_THREADS_H
#define FREE_TENSOR_GPU_NORMALIZE_THREADS_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class NormalizeThreads : public Mutator {
    struct IterInfo {
        std::string newIter_;
        Expr offset_;
    };

    Stmt root_;
    std::unordered_map<std::string, IterInfo> varMap_;
    std::unordered_map<ParallelScope, int>
        inside_; // Multiple nested `threadIdx.x`s are possible. See
                 // test/program/common_transforms::test_collaborative_fetch
    std::unordered_map<ParallelScope, std::vector<ID>> loops_;
    bool inKernel_ = false;

  public:
    NormalizeThreads(const Stmt &root) : root_(root) {}

  private:
    Stmt doVisitFor(const For &op);
    Stmt doVisitStmt(const Stmt &op);

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const Eval &op) override;
};

class CheckThreadNum
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> BaseClass;

    CompUniqueBounds bound_;

  public:
    CheckThreadNum() : bound_(*this, *this) {}

  private:
    /**
     * Ensure the length is defined with only constants and "byvalue" variables
     */
    bool isLegalLen(const Expr &expr);
    bool isLegalLen(const std::unordered_set<std::string> &names);

  protected:
    using BaseClass::visit;
    Stmt visit(const For &op) override;
};

Stmt normalizeThreads(const Stmt &op);

DEFINE_PASS_FOR_FUNC(normalizeThreads)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_NORMALIZE_THREADS_H
