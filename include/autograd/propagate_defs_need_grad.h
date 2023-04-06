#ifndef FREE_TENSOR_PROPAGATE_DEFS_NEED_GRAD_H
#define FREE_TENSOR_PROPAGATE_DEFS_NEED_GRAD_H

#include <unordered_set>

#include <analyze/symbol_table.h>
#include <container_utils.h>
#include <visitor.h>

namespace freetensor {

/**
 * Determine what variables we need to compute gradient for (propagete from
 * inputs to outputs)
 */
class PropagateRequires : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<ID> affectedDefs_; // all VarDef IDs

    ID curTarget_; // VarDef ID of current var being written to

  public:
    PropagateRequires(const std::unordered_set<std::string> &_requires,
                      const std::unordered_set<std::string> &provides)
        : requires_(_requires), provides_(provides) {}

    const std::unordered_set<ID> &affectedDefs() const { return affectedDefs_; }

    static std::unordered_set<ID>
    propagateUntilConverge(const Stmt &op,
                           const std::unordered_set<std::string> &_requires,
                           const std::unordered_set<std::string> &provides);

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

/**
 * Determine what variables we need to compute gradient for (propagete from
 * outputs to inputs)
 */
class PropagateProvides : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<ID> affectedDefs_; // all VarDef IDs

    ID curTarget_; // VarDef ID of current var being written to

  public:
    PropagateProvides(const std::unordered_set<std::string> &_requires,
                      const std::unordered_set<std::string> &provides)
        : requires_(_requires), provides_(provides) {}

    const std::unordered_set<ID> &affectedDefs() const { return affectedDefs_; }

    static std::unordered_set<ID>
    propagateUntilConverge(const Stmt &op,
                           const std::unordered_set<std::string> &_requires,
                           const std::unordered_set<std::string> &provides);

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

/**
 * To compute the required gradient, what intermediate gradients do we need?
 */
inline std::unordered_set<ID>
propagateDefsNeedGrad(const Stmt &op,
                      const std::unordered_set<std::string> &_requires,
                      const std::unordered_set<std::string> &provides) {
    return intersect(
        PropagateProvides::propagateUntilConverge(op, _requires, provides),
        PropagateRequires::propagateUntilConverge(op, _requires, provides));
}

} // namespace freetensor

#endif // FREE_TENSOR_PROPAGATE_DEFS_NEED_GRAD_H
