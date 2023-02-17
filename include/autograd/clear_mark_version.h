#ifndef FREE_TENSOR_CLEAR_MARK_VERSION_H
#define FREE_TENSOR_CLEAR_MARK_VERSION_H

#include <func.h>
#include <mutator.h>
#include <pass/flatten_stmt_seq.h>

namespace freetensor {

class ClearMarkVersion : public Mutator {
  protected:
    Stmt visit(const MarkVersion &op) { return makeStmtSeq({}); }
};

inline Stmt clearMarkVersion(const Stmt &op) {
    return flattenStmtSeq(ClearMarkVersion{}(op));
}

DEFINE_PASS_FOR_FUNC(clearMarkVersion)

} // namespace freetensor

#endif // FREE_TENSOR_CLEAR_MARK_VERSION_H
