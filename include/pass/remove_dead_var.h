#ifndef REMOVE_DEAD_VAR_H
#define REMOVE_DEAD_VAR_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace ir {

class RemoveAllWrites : public Mutator {
    std::string var_;

  public:
    RemoveAllWrites(const std::string &var) : var_(var) {}

  protected:
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class RemoveDeadVar : public Mutator {
    std::unordered_set<std::string> uses_;
    bool isFixPoint_ = true;

  public:
    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt removeDeadVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(removeDeadVar)

} // namespace ir

#endif // REMOVE_DEAD_VAR_H
