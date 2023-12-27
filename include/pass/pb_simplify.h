#ifndef FREE_TENSOR_PB_SIMPLIFY_H
#define FREE_TENSOR_PB_SIMPLIFY_H

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
#include <math/gen_pb_expr.h>
#include <math/parse_pb_expr.h>
#include <math/presburger.h>
#include <math/utils.h>
#include <pass/simplify.h>

namespace freetensor {

/**
 * CompUniqueBounds added with Presburger information
 *
 * For each statements in the AST, a corresponding instance of this class should
 * be created to deal with all (sub)expressions in the statement
 */
class CompUniqueBoundsPB : public CompUniqueBounds {
  public:
    class Bound : public CompUniqueBounds::Bound {
        Ref<PBCtx> ctx_;
        // isl var -> ft expr, the demangling map yielded from GenPBExpr
        // shared from CompUniqueBoundsPB::cachedFreeVars_
        Ref<std::unordered_map<std::string, Expr>> demangleMap_;
        // isl bounding set, multiple params being all outer variables and
        // single output being the bounded expression
        PBSet bound_;

        friend class CompUniqueBoundsPB;

      public:
        Bound(Ref<PBCtx> ctx,
              Ref<std::unordered_map<std::string, Expr>> demangleMap,
              PBSet bound)
            : ctx_(std::move(ctx)), demangleMap_(std::move(demangleMap)),
              bound_(std::move(bound)) {}

        int64_t lowerInt() const override;
        int64_t upperInt() const override;
        std::optional<int64_t> getInt() const override;

        Expr lowerExpr() const override;
        Expr upperExpr() const override;

        Ref<CompUniqueBounds::Bound> restrictScope(
            const std::unordered_set<std::string> &scope) const override;

        Expr simplestExpr(const std::unordered_map<std::string, int>
                              &orderedScope) const override;
    };

  private:
    const CompTransientBoundsInterface &transients_;
    GenPBExpr genPBExpr_;
    Ref<PBCtx> ctx_;

    Stmt cachedPlace_;
    PBSet cachedConds_;
    Ref<std::unordered_map<std::string, Expr>> cachedFreeVars_;
    std::unordered_map<Expr, Ref<Bound>> cachedValues_;

  public:
    CompUniqueBoundsPB(const CompTransientBoundsInterface &transients)
        : CompUniqueBounds(transients), transients_(transients),
          ctx_(Ref<PBCtx>::make()) {}

    Ref<CompUniqueBounds::Bound> getBound(const Expr &op) override;
    bool alwaysLE(const Expr &lhs, const Expr &rhs) override;
    bool alwaysLT(const Expr &lhs, const Expr &rhs) override;
    std::pair<Expr, Expr> unionBounds(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds) override;
};

class PBSimplify : public SimplifyPass {
  public:
    PBSimplify()
        : SimplifyPass([](const CompTransientBoundsInterface &tr) {
              return Ref<CompUniqueBoundsPB>::make(tr);
          }) {}
};

Stmt pbSimplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(pbSimplify)

} // namespace freetensor

#endif // FREE_TENSOR_PB_SIMPLIFY_H
