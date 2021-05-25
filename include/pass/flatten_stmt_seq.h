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
};

inline Stmt flattenStmtSeq(const Stmt &op, bool popVarDef = false) {
    return FlattenStmtSeq(popVarDef)(op);
}

inline Func flattenStmtSeq(const Func &func, bool popVarDef = false) {
    return makeFunc(func->params_, flattenStmtSeq(func->body_, popVarDef));
}

} // namespace ir

#endif // FLATTEN_STMT_SEQ_H
