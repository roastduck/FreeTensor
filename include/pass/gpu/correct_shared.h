#ifndef GPU_CORRECT_SHARED_H
#define GPU_CORRECT_SHARED_H

#include <unordered_map>
#include <vector>

#include <mutator.h>

namespace ir {

namespace gpu {

/**
 * Correct dimensions of shared memory buffers defined in local scope
 *
 * E.g. Alter from `shmem[i]` to `shmem[threadIdx.x, i]`
 *
 * NOTE: Do NOT call `shrinkVar` after this pass, or the result will be undone
 */
class CorrectShared : public Mutator {
    std::vector<For> stack_;
    std::unordered_map<std::string, int> defPos_;

    template <class T> T alterAccess(const T &op) {
        if (!defPos_.count(op->var_)) {
            return op;
        }
        int pos = defPos_.at(op->var_);
        for (int i = pos - 1; i >= 0; i--) {
            auto &indices = op->indices_;
            indices.insert(indices.begin(), makeVar(stack_[i]->iter_));
        }
        return op;
    }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

inline Stmt correctShared(const Stmt &op) { return CorrectShared()(op); }

} // namespace gpu

} // namespace ir

#endif // GPU_CORRECT_SHARED_H
