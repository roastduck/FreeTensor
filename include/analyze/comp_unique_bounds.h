#ifndef FREE_TENSOR_COMP_UNIQUE_BOUNDS_H
#define FREE_TENSOR_COMP_UNIQUE_BOUNDS_H

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>

namespace freetensor {

class CompUniqueBounds {
  public:
    enum class BoundType { Combination, Presburger };

    class Bound {
      protected:
        static int
        countScope(const Expr &op,
                   const std::unordered_map<std::string, int> &orderedScope);
        static int countHeavyOps(const Expr &op);

      public:
        virtual ~Bound() {}

        virtual BoundType type() const = 0;

        /**
         * Get an integer bound. In case of no solution, return LLONG_MAX or
         * LLONG_MIN
         *
         * @{
         */
        virtual int64_t lowerInt() const = 0;
        virtual int64_t upperInt() const = 0;
        /** @} */

        /**
         * If the bounded value is a constant integer, return it
         */
        virtual std::optional<int64_t> getInt() const = 0;

        /**
         * Return an Expr for the bound. In case of no solution, return nullptr
         *
         * @{
         */
        virtual Expr lowerExpr() const = 0;
        virtual Expr upperExpr() const = 0;
        /** @} */

        virtual Ref<Bound>
        restrictScope(const std::unordered_set<std::string> &scope) const = 0;

        virtual Expr simplestExpr(
            const Expr &reference,
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

} // namespace freetensor

#endif // FREE_TENSOR_COMP_UNIQUE_BOUNDS_H
