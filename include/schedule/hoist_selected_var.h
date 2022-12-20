#ifndef FREE_TENSOR_HOIST_SELECTED_VAR_H
#define FREE_TENSOR_HOIST_SELECTED_VAR_H

#include <analyze/symbol_table.h>
#include <mutator.h>
#include <selector.h>

namespace freetensor {

class HoistSelectedVar : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_set<ID> &toHoist_;

  public:
    HoistSelectedVar(const std::unordered_set<ID> &toHoist)
        : toHoist_(toHoist) {}

  protected:
    using BaseClass::visit;
    // We use unsual modify-before-recursion in this Mutator, and therefore
    // there is no need to hoist a VarDef over another VarDef, and we can keep
    // relative position of VarDef nodes
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const Assume &op) override;
    Stmt visit(const StmtSeq &op) override;
};

/**
 * Hoist all selected VarDef nodes untill they can no longer be selected by the
 * selector
 *
 * You only need to describe an area in the selector. No need to limit the node
 * type to be VarDef
 *
 * Algorithm: Each time we hoist all selected nodes over its direct parent (so
 * relative position of each VarDef nodes will be kept intact). Repeat until
 * convergance
 *
 * @param selector : Hoist all VarDef nodes in this area
 * @throw InvalidSchedule if the hoisting is impossible
 *
 * @{
 */
Stmt hoistSelectedVar(const Stmt &op, const std::string &selector);
Stmt hoistSelectedVar(const Stmt &op, const Ref<Selector> &selector);
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_HOIST_SELECTED_VAR_H
