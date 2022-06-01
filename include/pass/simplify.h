#ifndef FREE_TENSOR_SIMPLIFY_H
#define FREE_TENSOR_SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <math/bounds.h>
#include <mutator.h>
#include <opt.h>
#include <pass/annotate_conds.h>
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

// NOTE: We use ConstFold because we cannot rely the bound analysis for constant
// propagation. E.g f(x) + 0, where f(x) is a complex expression and it does not
// have a bound. The "+ 0" cannot be removed by bound analysis
class SimplifyPass : public CompTransientBounds<SymbolTable<ConstFold>> {
    typedef CompTransientBounds<SymbolTable<ConstFold>> BaseClass;

    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

    CompUniqueBounds &unique_;

  public:
    SimplifyPass(CompUniqueBounds &unique) : unique_(unique) {}

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

    Expr visitExpr(const Expr &op) override;

    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const RoundTowards0Div &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const LT &op) override;
    Expr visit(const LE &op) override;
    Expr visit(const GT &op) override;
    Expr visit(const GE &op) override;
    Expr visit(const EQ &op) override;
    Expr visit(const NE &op) override;
    Expr visit(const LAnd &op) override;
    Expr visit(const LOr &op) override;
    Expr visit(const LNot &op) override;
    Expr visit(const IfExpr &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

class BuiltinSimplify : public SimplifyPass {
    CompUniqueBounds unique_;

  public:
    BuiltinSimplify() : SimplifyPass(unique_), unique_(*this) {}
};

/**
 * Simplify a program and compute bounds of each expressions
 *
 * This pass can only be applied on a complete program, instead of a single
 * expression, because it examines VarDef nodes of each Var
 *
 * @return : {simplified, lower, upper}
 */
template <class Simplifier> Stmt simplifyImpl(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = annotateConds(op);
        auto newOp = Simplifier()(op);
        if (HashComparator()(newOp, op) || i > 100) {
            if (i > 100) {
                WARNING("SimplifyPass iterates over 100 rounds. Maybe there is "
                        "a bug");
            }
            return newOp;
        }
        op = newOp;
    }
}

Stmt builtinSimplify(const Stmt &op);

Stmt simplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(simplify)

} // namespace freetensor

#endif // FREE_TENSOR_SIMPLIFY_H
