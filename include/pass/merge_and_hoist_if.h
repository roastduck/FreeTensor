#ifndef MERGE_AND_HOIST_IF_H
#define MERGE_AND_HOIST_IF_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace ir {

/**
 * Merge and hoist If nodes
 *
 * 1. Merge consecutive If nodes with the same condition
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
 *
 * 2. Hoist loop-independent If nodes
 *
 * E.g. Transform
 *
 * ```
 * for i = 0 to n {
 *   if (n < 100) { ... }
 * }
 * ```
 *
 * into
 *
 * ```
 * if (n < 100) {
 *   for i = 0 to n { ... }
 * }
 * ```
 *
 * We do not handle else-cases in this pass. Otherwise, the resulting code will
 * be too long. This is different from pass/seperate_tail
 */
class MergeAndHoistIf : public Mutator {
    std::unordered_set<std::string> def_;

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
};

Stmt mergeAndHoistIf(const Stmt &op);

inline Func mergeAndHoistIf(const Func &func) {
    return makeFunc(func->name_, func->params_, func->buffers_, mergeAndHoistIf(func->body_),
                    func->src_);
}

} // namespace ir

#endif // MERGE_AND_HOIST_IF_H
