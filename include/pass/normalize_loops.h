#ifndef FREE_TENSOR_NORMALIZE_LOOPS_H
#define FREE_TENSOR_NORMALIZE_LOOPS_H

#include <functional>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

class NormalizeLoops : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::function<bool(const For &)> filter_;
    std::unordered_set<For> filteredIn_;

  public:
    NormalizeLoops(const std::function<bool(const For &)> &filter = nullptr)
        : filter_(filter) {}

  protected:
    using BaseClass::visit;
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
};

/**
 * Make loops to begin at 0 and have step 1
 *
 * @param filter : Optional. Normalize only filtered loops
 */
Stmt normalizeLoops(const Stmt &op,
                    const std::function<bool(const For &)> &filter = nullptr);

} // namespace freetensor

#endif // FREE_TENSOR_NORMALIZE_LOOPS_H
