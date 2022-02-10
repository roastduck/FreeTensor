#ifndef ANNOTATE_CONDS_H
#define ANNOTATE_CONDS_H

#include <mutator.h>

namespace ir {

class AnnotateConds : public Mutator {
    std::vector<Expr> conds_; // stack of conditions, may be null

  private:
    void addCond(const Expr &expr, bool negate = false);

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const Assume &op) override;
};

/**
 * Some conditions are implicit in If or Assert, this pass annotates the AST by
 * introducing extra Assume nodes
 *
 * This pass clears all previously set Assume nodes
 *
 * Currently, this pass annotates in the following situations
 *
 * 1. Conditions on R/W variables
 *
 * If the condition of an If or Assert node is related to an R/W variable, the
 * condition may not hold in all statments in the body. For example:
 *
 * ```
 * if (x < 4) {
 *   // stmt 1
 *   x += 1
 *   // stmt 2
 * }
 * ```
 *
 * The condition `x < 4` may no longer hold for `stmt 2`. Therefore, we annotate
 * the AST by transform it into
 *
 * ```
 * if (x < 4) {
 *   assume (x < 4) {
 *     // stmt 1
 *   }
 *   x += 1
 *   // stmt 2
 * }
 * ```
 */
inline Stmt annotateConds(const Stmt &op) { return AnnotateConds()(op); }

} // namespace ir

#endif // ANNOTATE_CONDS_H
