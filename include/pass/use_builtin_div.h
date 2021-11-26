#ifndef USE_BUILTIN_DIV_H
#define USE_BUILTIN_DIV_H

#include <func.h>
#include <pass/simplify.h>

namespace ir {

class UseBuiltinDiv : public CompUniqueBounds {
  protected:
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const Mod &op) override;
};

/**
 * Try to replace FloorDiv and CeilDiv with RoundTowards0Div
 */
Stmt useBuiltinDiv(const Stmt &op);

DEFINE_PASS_FOR_FUNC(useBuiltinDiv)

} // namespace ir

#endif // USE_BUILTIN_DIV_H
