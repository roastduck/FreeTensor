#ifndef COMP_ACCESS_BOUND_H
#define COMP_ACCESS_BOUND_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/bounds.h>
#include <mutator.h>

namespace ir {

class CompAccessBound : public Mutator {
    // bounds from AnalyzeBounds
    const std::unordered_map<Expr, std::vector<Bound>> &lower_;
    const std::unordered_map<Expr, std::vector<Bound>> &upper_;

    // var name -> [indices for each access]
    std::unordered_map<std::string, std::vector<std::vector<Expr>>> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

  public:
    CompAccessBound(const std::unordered_map<Expr, std::vector<Bound>> &lower,
                    const std::unordered_map<Expr, std::vector<Bound>> &upper)
        : lower_(lower), upper_(upper) {}

  private:
    static Expr reduceMin(const Expr &reduction, const Expr &item);
    static Expr reduceMax(const Expr &reduction, const Expr &item);

  protected:
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const For &op) override;
};

} // namespace ir

#endif // COMP_ACCESS_BOUND_H
