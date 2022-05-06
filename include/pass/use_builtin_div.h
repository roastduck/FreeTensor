#ifndef FREE_TENSOR_USE_BUILTIN_DIV_H
#define FREE_TENSOR_USE_BUILTIN_DIV_H

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class UseBuiltinDiv
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> BaseClass;

    CompUniqueBounds bound_;

  public:
    UseBuiltinDiv() : bound_(*this, *this) {}

  protected:
    using BaseClass::visit;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const Mod &op) override;
};

/**
 * Try to replace FloorDiv and CeilDiv with RoundTowards0Div
 */
Stmt useBuiltinDiv(const Stmt &op);

DEFINE_PASS_FOR_FUNC(useBuiltinDiv)

} // namespace freetensor

#endif // FREE_TENSOR_USE_BUILTIN_DIV_H
