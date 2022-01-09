#ifndef PB_SIMPLIFY_H
#define PB_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <pass/simplify.h>

namespace ir {

class PBCompBounds : public CompUniqueBounds {
    GenPBExpr genPBExpr_;
    PBCtx isl_;

  protected:
    using CompUniqueBounds::visit;

    Expr visitExpr(const Expr &op) override;
};

class PBSimplify : public SimplifyPass<PBCompBounds> {};

Stmt pbSimplify(const Stmt &op);

} // namespace ir

#endif // PB_SIMPLIFY_H
