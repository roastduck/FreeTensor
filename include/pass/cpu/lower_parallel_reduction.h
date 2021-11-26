#ifndef CPU_LOWER_PARALLEL_REDUCTION_H
#define CPU_LOWER_PARALLEL_REDUCTION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <func.h>
#include <mutator.h>

namespace ir {

namespace cpu {

class LowerParallelReduction : public Mutator {
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    std::vector<For> loopStack_;
    GetHash getHash_;

  private:
    uint64_t getHash(const Expr &op);

    std::vector<std::pair<For, int>> reducedBy(const ReduceTo &op);

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Although parallel reduction enjoys a native support by OpenMP, it does not
 * support using parallel reduction and atomic reduction simulteneously.
 * Therefore, we need to make some transformations
 */
inline Stmt lowerParallelReduction(const Stmt &op) {
    return LowerParallelReduction()(op);
}

DEFINE_PASS_FOR_FUNC(lowerParallelReduction)

} // namespace cpu

} // namespace ir

#endif // CPU_LOWER_PARALLEL_REDUCTION_H
