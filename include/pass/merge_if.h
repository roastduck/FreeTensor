#ifndef MERGE_IF_H
#define MERGE_IF_H

#include <mutator.h>

namespace ir {

/**
 * Merge consecutive If nodes with the same condition
 *
 * E.g. Transform
 *
 * ```
 * if (x) { f(); }
 * if (x) { g(); }
 * ```
 *
 * into
 *
 * ```
 * if (x) {
 *   f();
 *   g();
 * }
 * ```
 */
class MergeIf : public Mutator {
  protected:
    Stmt visit(const StmtSeq &op) override;
};

Stmt mergeIf(const Stmt &op);

} // namespace ir

#endif // MERGE_IF_H
