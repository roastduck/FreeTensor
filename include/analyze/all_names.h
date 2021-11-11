#ifndef ALL_NAMES_H
#define ALL_NAMES_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Record all Var and VarDef nodes that are used in an AST
 */
class AllNames : public Visitor {
    std::unordered_set<std::string> names_;

  public:
    const std::unordered_set<std::string> &names() const { return names_; }

  protected:
    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

std::unordered_set<std::string> allNames(const AST &op);

} // namespace ir

#endif // ALL_NAMES_H
