#ifndef GPU_NORMALIZE_THREADS_H
#define GPU_NORMALIZE_THREADS_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

namespace gpu {

class NormalizeThreads : public Mutator {
    struct IterInfo {
        std::string newIter_;
        Expr offset_;
    };

    std::unordered_map<std::string, IterInfo> varMap_;
    bool inKernel_ = false;

  private:
    Stmt doVisitFor(const For &op);

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
};

Stmt normalizeThreads(const Stmt &op);

} // namespace gpu

} // namespace ir

#endif // GPU_NORMALIZE_THREADS_H
