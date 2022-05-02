#ifndef FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H
#define FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H

#include <unordered_map>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class LowerParallelReduction : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::vector<For> loopStack_;

  private:
    std::vector<std::pair<For, int>> reducedBy(const ReduceTo &op);

  protected:
    using BaseClass::visit;
    Stmt visit(const For &op) override;
    Stmt visit(const ReduceTo &op) override;
};

Stmt lowerParallelReduction(const Stmt &op);

DEFINE_PASS_FOR_FUNC(lowerParallelReduction)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H
