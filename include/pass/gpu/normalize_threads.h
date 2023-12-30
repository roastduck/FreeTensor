#ifndef FREE_TENSOR_GPU_NORMALIZE_THREADS_H
#define FREE_TENSOR_GPU_NORMALIZE_THREADS_H

#ifdef FT_WITH_CUDA

#include <unordered_map>

#include <func.h>
#include <mutator.h>
#include <pass/shrink_for.h>

namespace freetensor {

namespace gpu {

class NormalizeThreads : public Mutator {
    Stmt root_;
    std::unordered_map<std::string, std::string> varMap_;
    std::unordered_map<ParallelScope, int>
        inside_; // Multiple nested `threadIdx.x`s are possible. See the
                 // doc-string of schedule/parallelize
    std::unordered_map<ParallelScope, std::vector<ID>> loops_;
    bool inKernel_ = false;

  public:
    NormalizeThreads(const Stmt &root) : root_(root) {}

  private:
    Stmt makeParallelScopes(const Stmt &body);

    Stmt doVisitFor(const For &op);
    Stmt doVisitStmt(const Stmt &op);

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const Eval &op) override;
};

class ShrinkNormalizedThreads : public ShrinkFor {
    typedef ShrinkFor BaseClass;

    std::unordered_set<For> openLoopsInKernel_;
    bool inKernel_ = false;

  protected:
    bool filterLoop(const For &op) override;

    std::unordered_set<std::string>
    filterNames(const std::unordered_set<std::string> &names) override;

  protected:
    using BaseClass::visit;
    Stmt visit(const For &op) override;
};

/**
 * Make GPU parallel scopes to be GPU kernel-like scopes
 *
 * After this pass, each kernel will be surrounded by perfectly nested block and
 * thread scopes. Semantic of the original program will be perserved by adding
 * conditions into the kernel body
 */
Stmt normalizeThreads(const Stmt &op);

DEFINE_PASS_FOR_FUNC(normalizeThreads)

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_NORMALIZE_THREADS_H
