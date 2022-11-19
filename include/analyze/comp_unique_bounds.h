#ifndef FREE_TENSOR_COMP_UNIQUE_BOUNDS_H
#define FREE_TENSOR_COMP_UNIQUE_BOUNDS_H

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <math/bounds.h>
#include <visitor.h>

namespace freetensor {

class CompUniqueBounds {
  public:
    class Bound {
      public:
        virtual ~Bound() {}

        virtual int64_t lowerInt() const = 0;
        virtual int64_t upperInt() const = 0;
        virtual std::optional<int64_t> getInt() const = 0;

        virtual Expr lowerExpr() const = 0;
        virtual Expr upperExpr() const = 0;

        virtual Ref<Bound>
        restrictScope(const std::unordered_set<std::string> &scope) const = 0;

        virtual Expr simplestExpr(
            const std::unordered_map<std::string, int> &orderedScope) const = 0;
    };

  protected:
    const CompTransientBoundsInterface &transients_;

  public:
    CompUniqueBounds(const CompTransientBoundsInterface &transients)
        : transients_(transients) {}
    virtual ~CompUniqueBounds() {}

    virtual Ref<Bound> getBound(const Expr &op) = 0;

    int64_t getIntLower(const Expr &op) { return getBound(op)->lowerInt(); }
    int64_t getIntUpper(const Expr &op) { return getBound(op)->upperInt(); }
    std::optional<int64_t> getInt(const Expr &op) {
        return getBound(op)->getInt();
    }

    virtual bool alwaysLT(const Expr &lhs, const Expr &rhs) = 0;
    virtual bool alwaysLE(const Expr &lhs, const Expr &rhs) = 0;

    virtual std::pair<Expr, Expr>
    unionBounds(const std::vector<Ref<Bound>> &bounds) = 0;
};

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
 * This pass is not accurate. Simplifying passes using this analysis may need
 * to run for multiple rounds
 */
class CompUniqueBoundsCombination : public CompUniqueBounds, public Visitor {
    typedef Visitor BaseClass;

    typedef std::vector<LowerBound> LowerBoundsList;
    typedef std::vector<UpperBound> UpperBoundsList;
    typedef std::unordered_map<Expr, LowerBoundsList> LowerBoundsMap;
    typedef std::unordered_map<Expr, UpperBoundsList> UpperBoundsMap;

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

    void updLower(LowerBoundsList &list, const LowerBound &bound) const;
    void updUpper(UpperBoundsList &list, const UpperBound &bound) const;

  private:
    /**
     * When analyzing Add, Sub and Mul, we first convert it to an linear
     * expression before analyzing bounds, so `a - a: l <= a <= r` results in `0
     * <= a - a <= 0`, instead of `l - r, l - a, a - r, 0 <= a - a <= r - l, r -
     * a, a - l, 0`
     */
    void visitLinear(const Expr &op);

  protected:
    void visitExpr(const Expr &op) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
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

#endif // FREE_TENSOR_COMP_UNIQUE_BOUNDS_H
