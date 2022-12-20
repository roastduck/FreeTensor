#ifndef FREE_TENSOR_SHRINK_FOR_H
#define FREE_TENSOR_SHRINK_FOR_H

#include <analyze/check_all_defined.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <container_utils.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <pass/pb_simplify.h>

namespace freetensor {

class CheckSideEffect : public Visitor {
    bool hasSideEffect_ = false;

  public:
    bool hasSideEffect() const { return hasSideEffect_; }

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Intrinsic &op) override;
};

class ShrinkFor : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

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

    Stmt subAST_;
    std::unordered_set<Stmt> subASTAncestors_;
    bool inSubAST_ = false;

  public:
    ShrinkFor() : bound_(*this) {}

    void setSubAST(const Stmt &subAST);

  protected:
    using BaseClass::visit;

    Stmt visitStmt(const Stmt &stmt) override;
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 */
Stmt shrinkFor(const Stmt &op, const Stmt &subAST = nullptr,
               bool doSimplify = true);

DEFINE_PASS_FOR_FUNC(shrinkFor)

} // namespace freetensor

#endif // FREE_TENSOR_SHRINK_FOR_H
