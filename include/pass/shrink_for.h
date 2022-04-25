#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <itertools.hpp>

#include <analyze/check_all_defined.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <pass/pb_simplify.h>

namespace ir {

class CheckSideEffect : public Visitor {
    bool hasSideEffect_ = false;

  public:
    bool hasSideEffect() const { return hasSideEffect_; }

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Intrinsic &op) override;
};

class ShrinkFor
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> BaseClass;

    // We need linear programming from PBCompBounds, because the minimum/maximum
    // value of a linear function does not always appear at the minimum/maximum
    // points of its parameters.
    // See 2.pass/test_shrink_for.py::test_linear_bounds
    PBCompBounds bound_;

    ASTHashMap<Var, std::pair<std::vector<std::vector<Expr>>,
                              std::vector<std::vector<Expr>>>>
        newRange_;
    std::vector<Var> iterStack_;
    std::vector<std::unordered_set<std::string>> namesStack_;

  public:
    ShrinkFor() : bound_(*this, *this) {}

  protected:
    using BaseClass::visit;

    Stmt visitStmt(const Stmt &stmt) override;
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 */
Stmt shrinkFor(const Stmt &op);

DEFINE_PASS_FOR_FUNC(shrinkFor)

} // namespace ir

#endif // SHRINK_FOR_H
