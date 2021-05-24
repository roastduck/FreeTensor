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
    Expr cond_;               // Conditions surrounding the accesses
};

typedef int CompAccessBoundMode;
const CompAccessBoundMode COMP_ACCESS_BOUND_READ = 0x1;
const CompAccessBoundMode COMP_ACCESS_BOUND_WRITE = 0x2;
const CompAccessBoundMode COMP_ACCESS_BOUND_ALL =
    COMP_ACCESS_BOUND_READ | COMP_ACCESS_BOUND_WRITE;

class FindMemType : public Visitor {
    std::string varDefId_;
    MemType mtype_;

  public:
    FindMemType(const std::string &varDefId) : varDefId_(varDefId) {}

    MemType mtype() const { return mtype_; }

  protected:
    void visit(const VarDef &op) override;
};

class CompAccessBound : public Visitor {
    struct Access {
        std::vector<Expr> indices_, conds_;

        Access(const std::vector<Expr> &indices, const std::vector<Expr> &conds)
            : indices_(indices), conds_(conds) {}
    };

    // The variable to compute
    std::string varDefId_;
    std::string var_;
    MemType mtype_;

    // bounds from AnalyzeBounds
    const std::unordered_map<Expr, std::vector<LowerBound>> &lower_;
    const std::unordered_map<Expr, std::vector<UpperBound>> &upper_;

    // each access to the specific variable
    std::vector<Access> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

    // all branching conditions
    std::vector<Expr> condStack_;

    CompAccessBoundMode mode_;

    AccessBound result_;

  public:
    CompAccessBound(
        const std::string &varDefId, MemType mtype,
        const std::unordered_map<Expr, std::vector<LowerBound>> &lower,
        const std::unordered_map<Expr, std::vector<UpperBound>> &upper,
        CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL)
        : varDefId_(varDefId), mtype_(mtype), lower_(lower), upper_(upper),
          mode_(mode) {}

    const AccessBound &result() const { return result_; }

  protected:
    void visit(const VarDef &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
};

AccessBound
compAccessBound(const Stmt &op, const std::string &varDefId,
                const std::unordered_map<Expr, std::vector<LowerBound>> &lower,
                const std::unordered_map<Expr, std::vector<UpperBound>> &upper,
                CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL);

} // namespace ir

#endif // COMP_ACCESS_BOUND_H
