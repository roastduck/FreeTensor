#ifndef FLATTEN_STMT_SEQ_H
#define FLATTEN_STMT_SEQ_H

#include <func.h>
#include <mutator.h>

namespace ir {

class FlattenStmtSeq : public Mutator {
    bool popVarDef_; // { A; VarDef { B; }} -> VarDef { A; B; }

  public:
    FlattenStmtSeq(bool popVarDef) : popVarDef_(popVarDef) {}

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const Assume &op) override;
};

/**
 * Merge nested StmtSeq nodes into one
 *
 * This pass also clears Assume nodes
 *
 * @param popVarDef : { A; VarDef { B; }} -> VarDef { A; B; }
 */
inline Stmt flattenStmtSeq(const Stmt &op, bool popVarDef = false) {
    return FlattenStmtSeq(popVarDef)(op);
}

DEFINE_PASS_FOR_FUNC(flattenStmtSeq)

} // namespace ir

#endif // FLATTEN_STMT_SEQ_H
