#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <math/bounds.h>
#include <mutator.h>
#include <opt.h>
#include <visitor.h>

namespace ir {

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

struct TransientBound {
    Expr expr_;
    std::vector<Expr> lower_, upper_;
};

class CompTransientBoundsInterface {
  public:
    virtual TransientBound transient(const Expr &op) const = 0;
    virtual const std::vector<Expr> &conds() const = 0;
};

/**
 * Compute bounds of IDENTICAL INTEGER (sub)expressions AT A POSITION in the AST
 *
 * E.g.
 *
 * ```
 * if (x <= 2) {
 *   ... = x + x; // <- AT THIS POSITION
 * }
 * ```
 *
 * At the position above, ALL TWO IDENTICAL EXPRESSIONS `x` have an upper bound
 * 2
 *
 * Inherit this pass to use it
 */
class CompTransientBounds : public WithTypeInfer<SymbolTable<Mutator>>,
                            public CompTransientBoundsInterface {
    typedef WithTypeInfer<SymbolTable<Mutator>> BaseClass;

    // Bounds related to certain expressions
    // Bounds in transients_ has already been recursed with (*this)(...)
    ASTHashMap<Expr, TransientBound> transients_;

    // Original bounds
    std::vector<Expr> conds_;

  public:
    TransientBound transient(const Expr &op) const override;
    const std::vector<Expr> &conds() const override { return conds_; }

  private:
    static Expr sub1(const Expr &op);
    static Expr add1(const Expr &op);

    void applyCond(const Expr &cond,
                   const std::unordered_set<std::string> &bodyAllWrites);

  protected:
    using BaseClass::visit; // Avoid hiding virtual functions

    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const Assume &op) override;
};

/**
 * Compute bounds of each UNIQUE INTEGER (sub)expression
 *
 * E.g.
 *
 * ```
 * if (x < 2) {
 *   ... = x;
 * }
 * ... = x;
 * ```
 *
 * Two UNIQUE expressions `x` have different upper bounds
 *
 * Inherit this pass to use it
 *
 * This pass is not accurate. Simplifying passes using this analysis may need
 * to run for multiple rounds
 */
class CompUniqueBounds : public WithTypeInfer<Visitor> {
    typedef WithTypeInfer<Visitor> BaseClass;

  public:
    typedef std::vector<LowerBound> LowerBoundsList;
    typedef std::vector<UpperBound> UpperBoundsList;
    typedef std::unordered_map<Expr, LowerBoundsList> LowerBoundsMap;
    typedef std::unordered_map<Expr, UpperBoundsList> UpperBoundsMap;

  private:
    const CompTransientBoundsInterface &transients_;

    LowerBoundsMap lower_;
    UpperBoundsMap upper_;

  public:
    CompUniqueBounds(const SymbolTableInterface &symbolTable,
                     const CompTransientBoundsInterface &transients)
        : WithTypeInfer<Visitor>(symbolTable), transients_(transients) {}

    LowerBoundsList getLower(const Expr &op) {
        (*this)(op);
        return lower_.at(op);
    }
    UpperBoundsList getUpper(const Expr &op) {
        (*this)(op);
        return upper_.at(op);
    }

    template <class T> void setLower(const Expr &op, T &&list) {
        lower_[op] = std::forward<T>(list);
    }
    template <class T> void setUpper(const Expr &op, T &&list) {
        upper_[op] = std::forward<T>(list);
    }

    void updLower(LowerBoundsList &list, const LowerBound &bound) const;
    void updUpper(UpperBoundsList &list, const UpperBound &bound) const;

    int getIntLower(const Expr &op);
    int getIntUpper(const Expr &op);
    Opt<int> getInt(const Expr &op);

    bool alwaysLT(const Expr &lhs, const Expr &rhs);
    bool alwaysLE(const Expr &lhs, const Expr &rhs);

    const LowerBoundsMap &lower() const { return lower_; }
    const UpperBoundsMap &upper() const { return upper_; }

  protected:
    void visitExpr(const Expr &op) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const Square &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const Mod &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const IfExpr &op) override;
};

template <class Unique> class SimplifyPass : public CompTransientBounds {
    typedef CompTransientBounds BaseClass;

    // We cannot rely the bound analysis for constant propagation.
    // E.g f(x) + 0, where f(x) is a complex expression and it does not have a
    // bound. The "+ 0" cannot be removed by bound analysis
    std::unordered_map<Expr, int64_t> constants_;

    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

    Unique unique_;

  public:
    SimplifyPass() : unique_(*this, *this) {}

    const Unique &uniqueBounds() const { return unique_; }

  protected:
    using BaseClass::visit;

    Expr visitExpr(const Expr &op) override;

    Expr visit(const IntConst &op) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const RoundTowards0Div &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Remainder &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
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

class BuiltinSimplify : public SimplifyPass<CompUniqueBounds> {};

/**
 * Simplify a program and compute bounds of each expressions
 *
 * This pass can only be applied on a complete program, instead of a single
 * expression, because it examines VarDef nodes of each Var
 *
 * @return : {simplified, lower, upper}
 */
template <class Simplifier>
std::tuple<Stmt, typename CompUniqueBounds::LowerBoundsMap,
           typename CompUniqueBounds::UpperBoundsMap>
simplifyAndGetBounds(const Stmt &op);

Stmt builtinSimplify(const Stmt &op);

Stmt simplifyPass(const Stmt &op);

DEFINE_PASS_FOR_FUNC(simplifyPass)

} // namespace ir

#endif // SIMPLIFY_H
