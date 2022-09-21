#ifndef FREE_TENSOR_REMOVE_DEAD_VAR_H
#define FREE_TENSOR_REMOVE_DEAD_VAR_H

#include <unordered_set>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class RemoveAllWrites : public Mutator {
    std::string var_;

  public:
    RemoveAllWrites(const std::string &var) : var_(var) {}

  protected:
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class RemoveDeadVar : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::unordered_set<std::string> uses_;
    std::string destination_;
    bool isFixPoint_ = true;

  public:
    bool isFixPoint() const { return isFixPoint_; }

  protected:
    using BaseClass::visit;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt removeDeadVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(removeDeadVar)

} // namespace freetensor

#endif // FREE_TENSOR_REMOVE_DEAD_VAR_H
