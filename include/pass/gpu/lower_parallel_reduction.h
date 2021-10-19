#ifndef GPU_LOWER_PARALLEL_REDUCTION_H
#define GPU_LOWER_PARALLEL_REDUCTION_H

#include <unordered_map>

#include <analyze/hash.h>
#include <func.h>
#include <mutator.h>

namespace ir {

namespace gpu {

class LowerParallelReduction : public Mutator {
    std::unordered_map<uint64_t, For> expr2for_; // hash of reducing expr -> for
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    GetHash getHash_;

  private:
    uint64_t getHash(const Expr &op);

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const ReduceTo &op) override;
};

inline Stmt lowerParallelReduction(const Stmt &op) {
    return LowerParallelReduction()(op);
}

DEFINE_PASS_FOR_FUNC(lowerParallelReduction)

} // namespace gpu

} // namespace ir

#endif // GPU_LOWER_PARALLEL_REDUCTION_H
