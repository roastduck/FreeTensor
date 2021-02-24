#ifndef COMP_ACCESS_BOUND_H
#define COMP_ACCESS_BOUND_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/bounds.h>
#include <visitor.h>

namespace ir {

struct AccessBound {
    std::vector<Expr> lower_; // lower_bound(access)
    std::vector<Expr> len_;   // upper_bound(access_i - access_j)
};

class CompAccessBound : public Visitor {
    // bounds from AnalyzeBounds
    const std::unordered_map<Expr, std::vector<Bound>> &lower_;
    const std::unordered_map<Expr, std::vector<Bound>> &upper_;

    // var name -> [indices for each access]
    std::unordered_map<std::string, std::vector<std::vector<Expr>>> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

    std::unordered_map<std::string, AccessBound> results_;

  public:
    CompAccessBound(const std::unordered_map<Expr, std::vector<Bound>> &lower,
                    const std::unordered_map<Expr, std::vector<Bound>> &upper)
        : lower_(lower), upper_(upper) {}

    const std::unordered_map<std::string, AccessBound> results() const {
        return results_;
    }

  private:
    static Expr reduceMin(const Expr &reduction, const Expr &item);
    static Expr reduceMax(const Expr &reduction, const Expr &item);

  protected:
    void visit(const VarDef &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
};

} // namespace ir

#endif // COMP_ACCESS_BOUND_H
