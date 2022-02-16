#ifndef COMP_UNIQUE_BOUNDS_H
#define COMP_UNIQUE_BOUNDS_H

#include <analyze/comp_transient_bounds.h>
#include <analyze/type_infer.h>
#include <visitor.h>

namespace ir {

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
 * Inherit this pass to use it
 *
 * This pass is not accurate. Simplifying passes using this analysis may need
 * to run for multiple rounds
 */
class CompUniqueBounds : public WithTypeInfer<Visitor> {
    typedef WithTypeInfer<Visitor> BaseClass;

  public:
    typedef std::vector<LowerBound> LowerBoundsList;
    typedef std::vector<UpperBound> UpperBoundsList;
    typedef std::unordered_map<Expr, LowerBoundsList> LowerBoundsMap;
    typedef std::unordered_map<Expr, UpperBoundsList> UpperBoundsMap;

  private:
    const CompTransientBoundsInterface &transients_;

    LowerBoundsMap lower_;
    UpperBoundsMap upper_;

  public:
    CompUniqueBounds(const SymbolTableInterface &symbolTable,
                     const CompTransientBoundsInterface &transients)
        : WithTypeInfer<Visitor>(symbolTable), transients_(transients) {}

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

    int getIntLower(const Expr &op);
    int getIntUpper(const Expr &op);
    Opt<int> getInt(const Expr &op);

    bool alwaysLT(const Expr &lhs, const Expr &rhs);
    bool alwaysLE(const Expr &lhs, const Expr &rhs);

    const LowerBoundsMap &lower() const { return lower_; }
    const UpperBoundsMap &upper() const { return upper_; }

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

} // namespace ir

#endif // COMP_UNIQUE_BOUNDS_H
