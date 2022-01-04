#ifndef PB_SIMPLIFY_H
#define PB_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <pass/simplify.h>

namespace ir {

/**
 * GenPBExpr specialized for handling external variables and bounds
 */
class GenPBExprSimplify : public GenPBExpr {
    std::unordered_map<Expr, std::unordered_set<std::string>> vars_;
    GetHash getHash_;
    Expr parent_ = nullptr;

  public:
    const std::unordered_set<std::string> &vars(const Expr &op) {
        return vars_[op];
    }

  protected:
    using GenPBExpr::visit;
    void visitExpr(const Expr &op) override;
    void visit(const Var &op) override;
    void visit(const Load &op) override;
};

class PBCompBounds : public CompUniqueBounds {
    GenPBExprSimplify genPBExpr_;
    PBCtx isl_;

  protected:
    using CompUniqueBounds::visit;

    Expr visitExpr(const Expr &op) override;
};

class PBSimplify : public SimplifyPass<PBCompBounds> {};

Stmt pbSimplify(const Stmt &op);

} // namespace ir

#endif // PB_SIMPLIFY_H
