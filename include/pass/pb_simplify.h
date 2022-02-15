#ifndef PB_SIMPLIFY_H
#define PB_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <pass/simplify.h>

namespace ir {

class PBCompBounds : public CompUniqueBounds {
    const CompTransientBoundsInterface &transients_;
    GenPBExpr genPBExpr_;
    PBCtx isl_;
    std::unordered_set<Expr> visited_;

  public:
    PBCompBounds(const SymbolTableInterface &symbolTable,
                 const CompTransientBoundsInterface &transients)
        : CompUniqueBounds(symbolTable, transients), transients_(transients),
          genPBExpr_(symbolTable) {}

  protected:
    using CompUniqueBounds::visit;

    void visitExpr(const Expr &op) override;
};

class PBSimplify : public SimplifyPass<PBCompBounds> {};

Stmt pbSimplify(const Stmt &op);

} // namespace ir

#endif // PB_SIMPLIFY_H
