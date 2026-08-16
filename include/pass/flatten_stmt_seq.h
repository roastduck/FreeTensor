#ifndef FREE_TENSOR_FLATTEN_STMT_SEQ_H
#define FREE_TENSOR_FLATTEN_STMT_SEQ_H

#include <func.h>
#include <mutator.h>
#include <subprocess.h>

namespace freetensor {

class FlattenStmtSeq : public Mutator {
    bool isEmptySeq(const Stmt &s);

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const Assume &op) override;
};

/**
 * Merge nested StmtSeq nodes into one
 *
 * Empty `VarDef` nodes that do not output, `For`, `If` or `Assert` nodes will
 * be removed
 *
 * This pass also clears Assume nodes (even if they are not empty)
 */
inline Stmt
flattenStmtSeq(const Stmt &op,
               const std::optional<bool> &asSubprocess = std::nullopt,
               const std::optional<double> &timeout = std::nullopt) {
    if (shouldRunInSubprocess(asSubprocess, timeout)) {
        return runPassSubprocess<Stmt>("flatten_stmt_seq", op, {}, asSubprocess,
                                       timeout);
    }
    return FlattenStmtSeq()(op);
}

DEFINE_PASS_FOR_FUNC(flattenStmtSeq)

} // namespace freetensor

#endif // FREE_TENSOR_FLATTEN_STMT_SEQ_H
