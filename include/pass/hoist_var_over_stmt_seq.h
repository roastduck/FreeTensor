#ifndef HOIST_VAR_OVER_STMT_SEQ
#define HOIST_VAR_OVER_STMT_SEQ

#include <unordered_map>

#include <func.h>
#include <mutator.h>

namespace ir {

class HoistVarOverStmtSeq : public Mutator {
    std::unordered_map<std::string, std::string>
        rename_; // old name -> new name
    bool isFixPoint_ = true;

  public:
    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const StmtSeq &op) override;
};

Stmt hoistVarOverStmtSeq(const Stmt &op);

inline Func hoistVarOverStmtSeq(const Func &func) {
    return makeFunc(func->name_, func->params_,
                    hoistVarOverStmtSeq(func->body_), func->src_);
}

} // namespace ir

#endif // HOIST_VAR_OVER_STMT_SEQ
