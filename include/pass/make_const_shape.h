#ifndef FREE_TENSOR_MAKE_CONST_SHAPE_H
#define FREE_TENSOR_MAKE_CONST_SHAPE_H

#include <func.h>
#include <mutator.h>
#include <pass/pb_simplify.h>

namespace freetensor {

/**
 * Some backends do not support local variables with dynamic shapes. This pass
 * relaxed selected shapes to constants
 *
 * The "byvalue" memory type is treated as constants
 */
class MakeConstShape : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    PBCompBounds unique_;
    const std::vector<MemType> &mtypes_;

  private:
    bool isConstOrByValue(const std::unordered_set<std::string> &names) const;
    bool isConstOrByValue(const Expr &x) const;

  public:
    MakeConstShape(const std::vector<MemType> &mtypes)
        : unique_(*this), mtypes_(mtypes) {}

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
};

Stmt makeConstShape(const Stmt &op, const std::vector<MemType> &mtypes);

DEFINE_PASS_FOR_FUNC(makeConstShape)

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_CONST_SHAPE_H
