#ifndef GPU_CORRECT_SHARED_AND_GLOBAL_H
#define GPU_CORRECT_SHARED_AND_GLOBAL_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

namespace gpu {

class FindAffectingLoops : public Visitor {
    std::unordered_set<std::string> loops_;             // ID
    std::unordered_map<std::string, std::string> defs_; // name -> ID
    MemType mode_;

    // VarDef ID -> For ID
    std::unordered_map<std::string, std::unordered_set<std::string>> results_;
    // Expr -> For ID
    const std::unordered_map<
        Expr, std::unordered_map<std::string, LoopVariability>> &variants_;

  public:
    FindAffectingLoops(
        const std::unordered_map<
            Expr, std::unordered_map<std::string, LoopVariability>> &variants,
        MemType mode)
        : mode_(mode), variants_(variants) {}

    const std::unordered_map<std::string, std::unordered_set<std::string>> &
    results() const {
        return results_;
    }

  private:
    template <class T> void visitMemWrite(const T &op) {
        Visitor::visit(op);
        if (defs_.count(op->var_)) {
            for (auto &&idx : op->indices_) {
                for (auto &&loop : loops_) {
                    if (isVariant(variants_, idx, loop)) {
                        results_[defs_.at(op->var_)].insert(loop);
                    }
                }
            }
            for (auto &&loop : loops_) {
                if (isVariant(variants_, op->expr_, loop)) {
                    results_[defs_.at(op->var_)].insert(loop);
                }
            }
        }
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
    void visit(const Store &op) override { visitMemWrite(op); }
    void visit(const ReduceTo &op) override { visitMemWrite(op); }
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
        &affecting_;
    MemType mode_;

  public:
    CorrectMutator(
        const std::unordered_map<std::string, std::unordered_set<std::string>>
            &affecting,
        MemType mode)
        : affecting_(affecting), mode_(mode) {}

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
