#ifndef FREE_TENSOR_COMP_UNIQUE_BOUNDS_COMBINATION_H
#define FREE_TENSOR_COMP_UNIQUE_BOUNDS_COMBINATION_H

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <hash.h>
#include <math/bounds.h>
#include <visitor.h>

namespace freetensor {

/**
 * Compute bounds of each UNIQUE INTEGER (sub)expression
 *
 * E.g.
 *
 * ```
 * if (x < 2) {
 *   ... = x;
 * }
 * ... = x;
 * ```
 *
 * Two UNIQUE expressions `x` have different upper bounds
 *
 * For each statements in the AST, a corresponding instance of this class should
 * be created to deal with all (sub)expressions in the statement, so as to
 * distinguish different `x` sites in the example above
 *
 * This pass is not accurate. Simplifying passes using this analysis may need
 * to run for multiple rounds
 */
class CompUniqueBoundsCombination : public CompUniqueBounds, public Visitor {
    typedef Visitor BaseClass;

    typedef std::vector<LowerBound> LowerBoundsList;
    typedef std::vector<UpperBound> UpperBoundsList;
    typedef ASTHashMap<Expr, LowerBoundsList> LowerBoundsMap;
    typedef ASTHashMap<Expr, UpperBoundsList> UpperBoundsMap;

    LowerBoundsMap lower_;
    UpperBoundsMap upper_;

  public:
    class Bound : public CompUniqueBounds::Bound {
        // retrieving expr from Lower/UpperBound requires to be mutable. fake it
        // here.
        mutable std::vector<LowerBound> lowerBounds_;
        mutable std::vector<UpperBound> upperBounds_;

        friend class CompUniqueBoundsCombination;

      public:
        Bound(std::vector<LowerBound> lowerBounds,
              std::vector<UpperBound> upperBounds)
            : lowerBounds_(std::move(lowerBounds)),
              upperBounds_(std::move(upperBounds)) {}

        BoundType type() const override { return BoundType::Combination; }

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

    CompUniqueBoundsCombination(const CompTransientBoundsInterface &transients)
        : CompUniqueBounds(transients) {}

    Ref<CompUniqueBounds::Bound> getBound(const Expr &op) override;

    /**
     * Check wheter `lhs` is always less than `rhs`
     *
     * This is a fast non-recursing function, which check the less-than relation
     * literally, without invoking CompUniqueBounds again, but maybe imprecise.
     * For precise comparison, please use `getLower` or `getUpper` on
     * `makeSub(lhs, rhs)`
     */
    bool alwaysLT(const Expr &lhs, const Expr &rhs) override;
    bool alwaysLE(const Expr &lhs, const Expr &rhs) override;

    std::pair<Expr, Expr> unionBounds(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds) override;

  protected:
    LowerBoundsList getLower(const Expr &op) {
        (*this)(op);
        return lower_.at(op);
    }
    UpperBoundsList getUpper(const Expr &op) {
        (*this)(op);
        return upper_.at(op);
    }

    template <class T> void setLower(const Expr &op, T &&list) {
        lower_[op] = std::forward<T>(list);
    }
    template <class T> void setUpper(const Expr &op, T &&list) {
        upper_[op] = std::forward<T>(list);
    }

    /**
     * Insert a new bound to a list of bounds. But if the new bound is a trivial
     * deduction of existing bounds in the list, it will not be inserted
     *
     * @{
     */
    void updLower(LowerBoundsList &list, const LowerBound &bound) const;
    void updUpper(UpperBoundsList &list, const UpperBound &bound) const;
    /** @} */

  private:
    /**
     * When analyzing Add, Sub and Mul, we first convert it to an linear
     * expression before analyzing bounds, so `a - a: l <= a <= r` results in `0
     * <= a - a <= 0`, instead of `l - r, l - a, a - r, 0 <= a - a <= r - l, r -
     * a, a - l, 0`
     */
    void visitLinear(const Expr &op);

    void insertSignDataTypeInfo(const Expr &op);

  protected:
    void visitExpr(const Expr &op) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const Cast &op) override;
    void visit(const Intrinsic &op) override;
    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const Square &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const Mod &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const IfExpr &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_COMP_UNIQUE_BOUNDS_COMBINATION_H
