#ifndef PROP_CONST_H
#define PROP_CONST_H

#include <func.h>
#include <mutator.h>

namespace ir {

class ReplaceLoads : public Mutator {
    const std::unordered_map<Load, Expr> &replacement_;

  public:
    ReplaceLoads(const std::unordered_map<Load, Expr> &replacement)
        : replacement_(replacement) {}

  protected:
    Expr visit(const Load &op) override;
};

/**
 * Propagate constants
 *
 * E.g. transform
 *
 * ```
 * x[0] = 1
 * y[0] = x[0]
 * ```
 *
 * into
 *
 * ```
 * x[0] = 1
 * y[0] = 1
 * ```
 */
Stmt propConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(propConst)

} // namespace ir

#endif // PROP_CONST_H
