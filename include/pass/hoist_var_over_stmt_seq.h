#ifndef FREE_TENSOR_HOIST_VAR_OVER_STMT_SEQ
#define FREE_TENSOR_HOIST_VAR_OVER_STMT_SEQ

#include <unordered_map>

#include <func.h>
#include <mutator.h>

namespace freetensor {

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

/**
 * Transform things like `VarDef { stmt0 VarDef { stmt1 }}` into `VarDef {
 * VarDef { stmt0 stmt1 }}`
 *
 * This is not a optimization pass. It is intended to used inside other passes
 * and make them simpler. It is suggest to run `sinkVar` after these passes to
 * revert the effect of `hoistVarOverStmtSeq`
 */
Stmt hoistVarOverStmtSeq(const Stmt &op);

DEFINE_PASS_FOR_FUNC(hoistVarOverStmtSeq)

} // namespace freetensor

#endif // FREE_TENSOR_HOIST_VAR_OVER_STMT_SEQ
