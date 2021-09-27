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

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const VarDef &op) override;
};

inline Stmt removeDeadVar(const Stmt &op) { return RemoveDeadVar()(op); }

inline Func removeDeadVar(const Func &func) {
    return makeFunc(func->name_, func->params_, func->buffers_, removeDeadVar(func->body_),
                    func->src_);
}

} // namespace ir

#endif // REMOVE_DEAD_VAR_H
