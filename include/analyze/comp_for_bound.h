#ifndef COMP_FOR_BOUND_H
#define COMP_FOR_BOUND_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/bounds.h>
#include <mutator.h>

namespace ir {

class CompForBound : public Mutator {
    // bounds from AnalyzeBounds
    const std::unordered_map<const ExprNode *, std::vector<Bound>> &lower_;
    const std::unordered_map<const ExprNode *, std::vector<Bound>> &upper_;

    // iter name -> all use point
    std::unordered_map<std::string, std::vector<Expr>> uses_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

    bool inCond_ = false;

  public:
    CompForBound(
        const std::unordered_map<const ExprNode *, std::vector<Bound>> &lower,
        const std::unordered_map<const ExprNode *, std::vector<Bound>> &upper)
        : lower_(lower), upper_(upper) {}

  private:
    static Expr reduceMin(const Expr &reduction, const Expr &item);
    static Expr reduceMax(const Expr &reduction, const Expr &item);

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

} // namespace ir

#endif // COMP_FOR_BOUND_H
