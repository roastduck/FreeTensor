#ifndef FREE_TENSOR_PB_SIMPLIFY_H
#define FREE_TENSOR_PB_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <pass/simplify.h>

namespace freetensor {

class PBCompBounds : public CompUniqueBounds {
    const CompTransientBoundsInterface &transients_;
    GenPBExpr genPBExpr_;
    PBCtx isl_;
    std::unordered_set<Expr> visited_;

  public:
    PBCompBounds(const CompTransientBoundsInterface &transients)
        : CompUniqueBounds(transients), transients_(transients) {}

  protected:
    using CompUniqueBounds::visit;

    void visitExpr(const Expr &op) override;
};

class PBSimplify : public SimplifyPass {
    PBCompBounds unique_;

  public:
    PBSimplify() : SimplifyPass(unique_), unique_(*this) {}
};

Stmt pbSimplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(pbSimplify)

} // namespace freetensor

#endif // FREE_TENSOR_PB_SIMPLIFY_H
