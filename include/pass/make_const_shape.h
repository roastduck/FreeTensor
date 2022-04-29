#ifndef FREE_TENSOR_MAKE_CONST_SHAPE_H
#define FREE_TENSOR_MAKE_CONST_SHAPE_H

#include <func.h>
#include <mutator.h>
#include <pass/pb_simplify.h>

namespace freetensor {

/**
 * Some backends do not support local variables with dynamic shapes. This pass
 * relaxed selected shapes to constants
 */
class MakeConstShape
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> BaseClass;

    PBCompBounds unique_;
    const std::vector<MemType> &mtypes_;

  public:
    MakeConstShape(const std::vector<MemType> &mtypes)
        : unique_(*this, *this), mtypes_(mtypes) {}

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
};

Stmt makeConstShape(const Stmt &op, const std::vector<MemType> &mtypes);

DEFINE_PASS_FOR_FUNC(makeConstShape)

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_CONST_SHAPE_H
