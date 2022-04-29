#ifndef FREE_TENSOR_FLATTEN_STMT_SEQ_H
#define FREE_TENSOR_FLATTEN_STMT_SEQ_H

#include <func.h>
#include <mutator.h>

namespace freetensor {

class FlattenStmtSeq : public Mutator {
  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const Assume &op) override;
};

/**
 * Merge nested StmtSeq nodes into one
 *
 * This pass also clears Assume nodes
 */
inline Stmt flattenStmtSeq(const Stmt &op) { return FlattenStmtSeq()(op); }

DEFINE_PASS_FOR_FUNC(flattenStmtSeq)

} // namespace freetensor

#endif // FREE_TENSOR_FLATTEN_STMT_SEQ_H
