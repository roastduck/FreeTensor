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

    // All reads occured from the current statement to the last statement
    std::unordered_set<std::string> readsAfterward_;

    // All writes occured in each statemtns
    std::unordered_map<ID, std::unordered_set<std::string>> writes_;

    std::unordered_set<std::string> writtenToOutput_;
    std::unordered_map<std::string, int> inLoopCnt_;
    std::string destination_;
    bool isFixPoint_ = true;

  public:
    bool isFixPoint() const { return isFixPoint_; }

  protected:
    using BaseClass::visit;
    Stmt visitStmt(const Stmt &s) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

Stmt removeDeadVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(removeDeadVar)

} // namespace freetensor

#endif // FREE_TENSOR_REMOVE_DEAD_VAR_H
