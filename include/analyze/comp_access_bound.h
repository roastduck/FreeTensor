#ifndef COMP_ACCESS_BOUND_H
#define COMP_ACCESS_BOUND_H

#include <unordered_map>
#include <unordered_set>

#include <math/bounds.h>
#include <visitor.h>

namespace ir {

struct AccessBound {
    std::vector<Expr> lower_; // lower_bound(access)
    std::vector<Expr> len_;   // upper_bound(access_i - access_j)
};

typedef int CompAccessBoundMode;
const CompAccessBoundMode COMP_ACCESS_BOUND_READ = 0x1;
const CompAccessBoundMode COMP_ACCESS_BOUND_WRITE = 0x2;
const CompAccessBoundMode COMP_ACCESS_BOUND_ALL =
    COMP_ACCESS_BOUND_READ | COMP_ACCESS_BOUND_WRITE;

class CompAccessBound : public Visitor {
    // bounds from AnalyzeBounds
    const std::unordered_map<Expr, std::vector<LowerBound>> &lower_;
    const std::unordered_map<Expr, std::vector<UpperBound>> &upper_;

    // var name -> [indices for each access]
    std::unordered_map<std::string, std::vector<std::vector<Expr>>> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

    CompAccessBoundMode mode_;

    std::unordered_map<std::string, AccessBound> results_;

  public:
    CompAccessBound(
        const std::unordered_map<Expr, std::vector<LowerBound>> &lower,
        const std::unordered_map<Expr, std::vector<UpperBound>> &upper,
        CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL)
        : lower_(lower), upper_(upper), mode_(mode) {}

    const std::unordered_map<std::string, AccessBound> &results() const {
        return results_;
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
};

} // namespace ir

#endif // COMP_ACCESS_BOUND_H
