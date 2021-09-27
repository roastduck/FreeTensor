#ifndef MAKE_CONST_SHAPE_H
#define MAKE_CONST_SHAPE_H

#include <func.h>
#include <mutator.h>
#include <pass/simplify.h>

namespace ir {

/**
 * Some backends do not support local variables with dynamic shapes. This pass
 * relaxed selected shapes to constants
 */
class MakeConstShape : public Mutator {
    const std::vector<MemType> &mtypes_;
    const BuiltinSimplify::UpperBoundsMap &upper_;

  public:
    MakeConstShape(const std::vector<MemType> &mtypes,
                   const BuiltinSimplify::UpperBoundsMap &upper)
        : mtypes_(mtypes), upper_(upper) {}

  protected:
    Stmt visit(const VarDef &op) override;
};

Stmt makeConstShape(const Stmt &op, const std::vector<MemType> &mtypes);

inline Func makeConstShape(const Func &func,
                           const std::vector<MemType> &mtypes) {
    return makeFunc(func->name_, func->params_, func->buffers_,
                    makeConstShape(func->body_, mtypes), func->src_);
}

} // namespace ir

#endif // MAKE_CONST_SHAPE_H
