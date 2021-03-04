#ifndef USE_BUILTIN_DIV_H
#define USE_BUILTIN_DIV_H

#include <pass/simplify.h>

namespace ir {

class UseBuiltinDiv : public CompUniqueBounds {
  protected:
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
};

/**
 * Try to replace FloorDiv and CeilDiv with RoundTowards0Div
 */
Stmt useBuiltinDiv(const Stmt &op);

} // namespace ir

#endif // USE_BUILTIN_DIV_H
