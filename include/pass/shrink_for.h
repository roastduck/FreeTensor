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

    ASTHashMap<Var, std::vector<Ref<CompUniqueBounds::Bound>>> newRange_;
    std::vector<Var> iterStack_;
    std::vector<std::unordered_set<std::string>> namesStack_;

    Stmt subAST_;
    std::unordered_set<Stmt> subASTAncestors_;
    bool inSubAST_ = false;

  public:
    void setSubAST(const Stmt &subAST);

  protected:
    virtual bool filterLoop(const For &op) { return true; }

    virtual std::unordered_set<std::string>
    filterNames(const std::unordered_set<std::string> &names) {
        return names;
    }

  protected:
    using BaseClass::visit;

    Stmt visitStmt(const Stmt &stmt) override;
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 *
 * @param op : The AST to transform.
 * @param subAST : If specified, only transform sub-tree of the statement with
 * the specified ID.
 * @param doSimplify : If true, run simplify before and after the tranformation.
 * Transformations are required to ensure the effectiveness of the shrinking.
 * Please do your own simplification if you want to set it to false.
 *
 * @{
 */
Stmt shrinkFor(const Stmt &op, const ID &subAST = ID(), bool doSimplify = true);
inline Stmt shrinkFor(const Stmt &op, const Stmt &subAST,
                      bool doSimplify = true) {
    return shrinkFor(op, subAST.isValid() ? subAST->id() : ID(), doSimplify);
}
/** @} */

DEFINE_PASS_FOR_FUNC(shrinkFor)

} // namespace freetensor

#endif // FREE_TENSOR_SHRINK_FOR_H
