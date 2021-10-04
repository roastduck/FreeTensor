#ifndef PROP_CONST_H
#define PROP_CONST_H

#include <func.h>
#include <mutator.h>

namespace ir {

class ReplaceUses : public Mutator {
    const std::unordered_map<Load, Expr> &replaceLoad_;
    const std::unordered_map<ReduceTo, Expr> &replaceReduceTo_;

  public:
    ReplaceUses(const std::unordered_map<Load, Expr> &replaceLoad,
                const std::unordered_map<ReduceTo, Expr> &replaceReduceTo)
        : replaceLoad_(replaceLoad), replaceReduceTo_(replaceReduceTo) {}

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
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
