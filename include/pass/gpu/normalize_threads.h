#ifndef GPU_NORMALIZE_THREADS_H
#define GPU_NORMALIZE_THREADS_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>

namespace ir {

namespace gpu {

class NormalizeThreads : public Mutator {
    struct IterInfo {
        std::string newIter_;
        Expr offset_;
    };

    std::unordered_map<std::string, IterInfo> varMap_;
    std::unordered_set<std::string> inside_;
    bool inKernel_ = false;

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

Stmt normalizeThreads(const Stmt &op);

} // namespace gpu

} // namespace ir

#endif // GPU_NORMALIZE_THREADS_H
