#ifndef GPU_CORRECT_SHARED_AND_GLOBAL_H
#define GPU_CORRECT_SHARED_AND_GLOBAL_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

namespace gpu {

class FindParallelLoops : public Visitor {
    std::vector<For> loops_, stack_;
    std::unordered_map<std::string, std::unordered_set<std::string>> affecting_;

  public:
    const std::vector<For> &loops() const { return loops_; }
    const std::unordered_map<std::string, std::unordered_set<std::string>> &
    affecting() const {
        return affecting_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
};

/**
 * Correct dimensions of shared memory buffers defined in local scope
 *
 * E.g. Alter from `shmem[i]` to `shmem[threadIdx.x, i]`
 */
class CorrectMutator : public Mutator {
    std::vector<For> stack_;
    std::unordered_map<std::string, int> defPos_;
    std::unordered_map<std::string, std::string> defs_; // name -> ID
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        &affecting_; // VarDef ID -> For ID

  public:
    CorrectMutator(
        const std::unordered_map<std::string, std::unordered_set<std::string>>
            &affecting)
        : affecting_(affecting) {}

  private:
    template <class T> T alterAccess(const T &op) {
        if (!defPos_.count(op->var_)) {
            return op;
        }
        if (affecting_.count(defs_.at(op->var_))) {
            auto &&aff = affecting_.at(defs_.at(op->var_));
            int pos = defPos_.at(op->var_);
            for (int i = pos - 1; i >= 0; i--) {
                if (aff.count(stack_[i]->id())) {
                    auto &indices = op->indices_;
                    indices.insert(indices.begin(), makeVar(stack_[i]->iter_));
                }
            }
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

Stmt correctSharedAndGlobal(const Stmt &op);

inline Func correctSharedAndGlobal(const Func &func) {
    return makeFunc(func->name_, func->params_,
                    correctSharedAndGlobal(func->body_), func->src_);
}

} // namespace gpu

} // namespace ir

#endif // GPU_CORRECT_SHARED_AND_GLOBAL_H
