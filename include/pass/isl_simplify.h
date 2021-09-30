#ifndef ISL_SIMPLIFY_H
#define ISL_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <math/gen_isl_expr.h>
#include <math/isl.h>
#include <pass/simplify.h>

namespace ir {

/**
 * GenISLExpr specialized for handling external variables and bounds
 */
class GenISLExprSimplify : public GenISLExpr {
    std::unordered_map<Expr, std::unordered_set<std::string>> vars_;
    std::unordered_map<Expr, std::vector<std::string>> cond_;
    GetHash getHash_;
    Expr parent_ = nullptr;

  public:
    std::unordered_set<std::string> &vars(const Expr &op) { return vars_[op]; }
    std::vector<std::string> &cond(const Expr &op) { return cond_[op]; }

  protected:
    using GenISLExpr::visit;
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;
    void visit(const Var &op) override;
    void visit(const Load &op) override;
};

class ISLCompBounds : public CompUniqueBounds {
    GenISLExprSimplify genISLExpr_;
    ISLCtx isl_;

  protected:
    using CompUniqueBounds::visit;

    Expr visitExpr(const Expr &op,
                   const std::function<Expr(const Expr &)> &visitNode) override;
};

class ISLSimplify : public SimplifyPass<ISLCompBounds> {};

Stmt islSimplify(const Stmt &op);

} // namespace ir

#endif // ISL_SIMPLIFY_H
