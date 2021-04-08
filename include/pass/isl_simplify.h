#ifndef ISL_SIMPLIFY_H
#define ISL_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <isl/ctx.h>
#include <isl/ilp.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include <analyze/hash.h>
#include <pass/simplify.h>

namespace ir {

struct ISLExpr {
    std::unordered_set<int> var_;
    std::vector<std::string> cond_;
    std::string expr_;
};

class ISLCompBounds : public CompUniqueBounds {
    int varCnt_ = 0;
    std::unordered_map<uint64_t, int> varId_;
    GetHash getHash_;

    std::unordered_map<Expr, ISLExpr> islExprs_;

    isl_ctx *isl_;

  public:
    ISLCompBounds();
    ~ISLCompBounds();

  private:
    int getVarId(const Expr &op);

  protected:
    using CompUniqueBounds::visit;

    Expr visitExpr(const Expr &op,
                   const std::function<Expr(const Expr &)> &visitNode) override;

    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
    Expr visit(const IntConst &op) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
};

class ISLSimplify : public SimplifyPass<ISLCompBounds> {};

Stmt islSimplify(const Stmt &op);

} // namespace ir

#endif // ISL_SIMPLIFY_H
