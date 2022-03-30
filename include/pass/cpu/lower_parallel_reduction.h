#ifndef CPU_LOWER_PARALLEL_REDUCTION_H
#define CPU_LOWER_PARALLEL_REDUCTION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace ir {

namespace cpu {

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

/**
 * Although parallel reduction enjoys a native support by OpenMP, it does not
 * support using parallel reduction and atomic reduction simulteneously.
 * Therefore, we need to make some transformations
 */
Stmt lowerParallelReduction(const Stmt &op);

DEFINE_PASS_FOR_FUNC(lowerParallelReduction)

} // namespace cpu

} // namespace ir

#endif // CPU_LOWER_PARALLEL_REDUCTION_H
