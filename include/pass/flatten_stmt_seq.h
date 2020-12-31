#ifndef FLATTEN_STMT_SEQ_H
#define FLATTEN_STMT_SEQ_H

#include <mutator.h>

namespace ir {

class FlattenStmtSeq : public Mutator {
  protected:
    Stmt visit(const StmtSeq &op) override;
};

inline Stmt flattenStmtSeq(const Stmt &op) { return FlattenStmtSeq()(op); }

} // namespace ir

#endif // FLATTEN_STMT_SEQ_H
