#ifndef COMP_ACCESS_BOUND_H
#define COMP_ACCESS_BOUND_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
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
    ID varDefId_;
    MemType mtype_;

  public:
    FindMemType(const ID &varDefId) : varDefId_(varDefId) {}

    MemType mtype() const { return mtype_; }

  protected:
    void visit(const VarDef &op) override;
};

class CompAccessBound
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Visitor>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Visitor>>> BaseClass;

    struct Access {
        std::vector<Expr> indices_, conds_;
        std::vector<std::vector<LowerBound>> lower_;
        std::vector<std::vector<UpperBound>> upper_;

        Access(CompUniqueBounds &unique, const std::vector<Expr> &indices,
               const std::vector<Expr> &conds)
            : indices_(indices), conds_(conds) {
            for (auto &&idx : indices) {
                lower_.emplace_back(unique.getLower(idx));
                upper_.emplace_back(unique.getUpper(idx));
            }
        }
    };

    CompUniqueBounds unique_;

    // The variable to compute
    ID varDefId_;
    std::string var_;
    MemType mtype_;

    // each access to the specific variable
    std::vector<Access> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;

    CompAccessBoundMode mode_;

    AccessBound result_;

  public:
    CompAccessBound(const ID &varDefId, MemType mtype,
                    CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL)
        : unique_(*this, *this), varDefId_(varDefId), mtype_(mtype),
          mode_(mode) {}

    const AccessBound &result() const { return result_; }

  protected:
    using BaseClass::visit;
    void visit(const VarDef &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
};

AccessBound compAccessBound(const Stmt &op, const ID &varDefId,
                            CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL);

} // namespace ir

#endif // COMP_ACCESS_BOUND_H
