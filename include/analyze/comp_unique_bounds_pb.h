#ifndef FREE_TENSOR_COMP_UNIQUE_BOUNDS_PB_H
#define FREE_TENSOR_COMP_UNIQUE_BOUNDS_PB_H

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
#include <math/gen_pb_expr.h>
#include <math/presburger.h>

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
      public: // Visible to CompUniqueBoundsPB's subclasses
        Ref<PBCtx> ctx_;
        // isl var -> ft expr, the demangling map yielded from GenPBExpr
        // shared from CompUniqueBoundsPB::cachedFreeVars_
        Ref<std::unordered_map<std::string, Expr>> demangleMap_;
        // isl bounding set, multiple params being all outer variables and
        // single output being the bounded expression
        PBSet bound_;

      public:
        Bound(Ref<PBCtx> ctx,
              Ref<std::unordered_map<std::string, Expr>> demangleMap,
              PBSet bound)
            : ctx_(std::move(ctx)), demangleMap_(std::move(demangleMap)),
              bound_(std::move(bound)) {}

        BoundType type() const override { return BoundType::Presburger; }

        int64_t lowerInt() const override;
        int64_t upperInt() const override;
        std::optional<int64_t> getInt() const override;

        Expr lowerExpr() const override;
        Expr upperExpr() const override;

        Ref<CompUniqueBounds::Bound> restrictScope(
            const std::unordered_set<std::string> &scope) const override;

        Expr simplestExpr(const Expr &reference,
                          const std::unordered_map<std::string, int>
                              &orderedScope) const override;
    };

  private:
    const CompTransientBoundsInterface &transients_;
    GenPBExpr genPBExpr_;
    Ref<PBCtx> ctx_;

    PBSet cachedConds_;
    Ref<std::unordered_map<std::string, Expr>> cachedFreeVars_;
    std::unordered_map<Expr, Ref<Bound>> cachedValues_;

  protected:
    Ref<CompUniqueBoundsPB::Bound>
    unionBoundsAsBound(const std::vector<Ref<CompUniqueBounds::Bound>> &bounds);

  public:
    CompUniqueBoundsPB(const CompTransientBoundsInterface &transients);

    Ref<CompUniqueBounds::Bound> getBound(const Expr &op) override;
    bool alwaysLE(const Expr &lhs, const Expr &rhs) override;
    bool alwaysLT(const Expr &lhs, const Expr &rhs) override;
    std::pair<Expr, Expr> unionBounds(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_COMP_UNIQUE_BOUNDS_PB_H
