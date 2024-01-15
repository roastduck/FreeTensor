#ifndef FREE_TENSOR_SIMPLIFY_H
#define FREE_TENSOR_SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/comp_unique_bounds_combination.h>
#include <analyze/comp_unique_bounds_pb.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <math/bounds.h>
#include <mutator.h>
#include <pass/const_fold.h>
#include <visitor.h>

namespace freetensor {

/**
 * Find all the variables in an expression, and determine the inner most scope
 * where these variables are defined
 */
class FindInnerMostScope : public Visitor {
    const std::unordered_map<std::string, int> &varScope_;
    int innerMost_ = 0;

  public:
    FindInnerMostScope(const std::unordered_map<std::string, int> &varScope)
        : varScope_(varScope) {}
    int innnerMost() const { return innerMost_; }

  protected:
    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
};

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op);

/**
 * Base class for integer simplify passes
 *
 * Simplification from 3 mechanisms:
 *
 * - Bound analysis from a specific subclass of `CompUniqueBounds`. If there is
 *   only one integer between an expression's lower bound and upper bound, then
 *   the expression can be replaced by the integer.
 * - Constant folding from `ConstFold`.
 * - Simplification rules from `SimplifyPass`. This is a complement of the bound
 *   analysis. E.g. to simplify `x + 0` as `x`, which cannot be simplified by
 *   bound analysis if `x` has no bound.
 *
 * @param compUniqueBoundsFactor : A factory function creating a specific
 * `CompUniqueBounds` instance for bound analysis.
 * @param leafFirstBoundAnalysis : Whether to simplify sub-expressions with
 * bound analysis before simplifying their parents. This is useful when the
 * simplification of a sub-expression helps analyzing its parent, but will only
 * waste time if the analysis is based on a unified representation and does not
 * depend on the specific form of the sub-expression.
 */
class SimplifyPass : public CompTransientBounds<SymbolTable<ConstFold>> {
    typedef CompTransientBounds<SymbolTable<ConstFold>> BaseClass;

    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

    Ref<CompUniqueBounds> unique_;
    std::function<Ref<CompUniqueBounds>(const CompTransientBoundsInterface &)>
        compUniqueBoundsFactory_;
    bool leafFirstBoundAnalysis_;

  public:
    SimplifyPass(std::function<Ref<CompUniqueBounds>(
                     const CompTransientBoundsInterface &)>
                     compUniqueBoundsFactory,
                 bool leafFirstBoundAnalysis)
        : compUniqueBoundsFactory_(compUniqueBoundsFactory),
          leafFirstBoundAnalysis_(leafFirstBoundAnalysis) {}

  private:
    template <class T> bool equals(const Expr &op, T &&val) const {
        if (op->nodeType() == ASTNodeType::IntConst &&
            op.as<IntConstNode>()->val_ == val) {
            return true;
        }
        if (op->nodeType() == ASTNodeType::FloatConst &&
            op.as<FloatConstNode>()->val_ == val) {
            return true;
        }
        return false;
    }

  protected:
    using BaseClass::visit;

    Stmt visitStmt(const Stmt &op) override;
    Expr visitExpr(const Expr &op) override;

    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const RoundTowards0Div &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Remainder &op) override;
    Expr visit(const LT &op) override;
    Expr visit(const LE &op) override;
    Expr visit(const GT &op) override;
    Expr visit(const GE &op) override;
    Expr visit(const EQ &op) override;
    Expr visit(const NE &op) override;
    Expr visit(const IfExpr &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

class BuiltinSimplify : public SimplifyPass {
  public:
    BuiltinSimplify()
        : SimplifyPass(
              [](const CompTransientBoundsInterface &tr) {
                  return Ref<CompUniqueBoundsCombination>::make(tr);
              },
              true) {}
};

class PBSimplify : public SimplifyPass {
  public:
    PBSimplify()
        : SimplifyPass(
              [](const CompTransientBoundsInterface &tr) {
                  return Ref<CompUniqueBoundsPB>::make(tr);
              },
              false) {}
};

/**
 * Simplify integer expressions in a program
 *
 * `builtinSimplify` and `simplify` uses `CompUniqueBoundsCombination` to
 * simplify the program. `pbSimplify` uses `CompUniqueBoundsPB` to simplify the
 * program.
 *
 * This pass can only be applied on a complete program, instead of a single
 * expression, because it examines VarDef nodes of each Var
 *
 * @return : The simplified AST
 *
 * @{
 */
Stmt builtinSimplify(const Stmt &op);
Stmt pbSimplify(const Stmt &op);
Stmt simplify(const Stmt &op);
/** @} */

DEFINE_PASS_FOR_FUNC(builtinSimplify)
DEFINE_PASS_FOR_FUNC(pbSimplify)
DEFINE_PASS_FOR_FUNC(simplify)

} // namespace freetensor

#endif // FREE_TENSOR_SIMPLIFY_H
