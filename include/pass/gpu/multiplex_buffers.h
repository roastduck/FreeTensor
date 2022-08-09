#ifndef FREE_TENSOR_GPU_MULTIPLEX_BUFFERS_H
#define FREE_TENSOR_GPU_MULTIPLEX_BUFFERS_H

#ifdef FT_WITH_CUDA

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/symbol_table.h>
#include <driver/target.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

namespace gpu {

class FindParallelLoops : public Visitor {
    Ref<GPUTarget> target_;
    std::vector<For> loops_, stack_;
    std::unordered_map<ID, std::unordered_set<ID>> affecting_;

  public:
    FindParallelLoops(const Ref<GPUTarget> &target) : target_(target) {}

    const std::vector<For> &loops() const { return loops_; }
    const std::unordered_map<ID, std::unordered_set<ID>> &affecting() const {
        return affecting_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
};

class MultiplexMutator : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::vector<For> stack_;
    std::unordered_map<std::string, int> defPos_;
    const std::unordered_map<ID, std::unordered_set<ID>>
        &affecting_; // VarDef ID -> For ID

  public:
    MultiplexMutator(
        const std::unordered_map<ID, std::unordered_set<ID>> &affecting)
        : affecting_(affecting) {}

  private:
    template <class T> T alterAccess(const T &op) {
        if (!defPos_.count(op->var_)) {
            return op;
        }
        if (affecting_.count(def(op->var_)->id())) {
            auto &&aff = affecting_.at(def(op->var_)->id());
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

/**
 * If a shared or global VarDef is inside a parallel For region, it should be
 * enlarged so that each thread or block will access different parts of it
 *
 * E.g. Alter from `shmem[i]` to `shmem[threadIdx.x, i]`
 */
Stmt multiplexBuffers(const Stmt &op, const Ref<GPUTarget> &target);

DEFINE_PASS_FOR_FUNC(multiplexBuffers)

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_MULTIPLEX_BUFFERS_H
