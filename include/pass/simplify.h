#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/analyze_linear.h>
#include <analyze/type_infer.h>
#include <math/bounds.h>
#include <mutator.h>
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

class CheckBoundOutDated : public Visitor {
    std::string name_;
    bool outDated_;

  public:
    CheckBoundOutDated() : name_(""), outDated_(false) {}
    CheckBoundOutDated(const std::string &name)
        : name_(name), outDated_(false) {}
    void reset() { outDated_ = false; }
    bool isoutDated() { return outDated_; }

  protected:
    void visit(const Var &op) override;
    void visit(const Load &op) override;
};

struct TransientBound {
    Expr expr_;
    std::vector<Expr> lower_, upper_;
};

class OutDatedBoundsRemover : public Visitor {
    std::unordered_map<uint64_t, TransientBound> &transients_;
    CheckBoundOutDated check_;

  public:
    OutDatedBoundsRemover(
        std::unordered_map<uint64_t, TransientBound> &transients)
        : transients_(transients), check_() {}

  private:
    void remove(const std::string &name);

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

/**
 * Compute bounds of IDENTICAL (sub)expressions AT A POSITION in the AST
 *
 * E.g.
 *
 * ```
 * if (x < 2) {
 *   ... = x + x; // <- AT THIS POSITION
 * }
 * ```
 *
 * At the position above, ALL TWO IDENTICAL EXPRESSIONS `x` have an upper bound
 * 2
 *
 * Inherit this pass to use it
 */
class CompTransientBounds : public Mutator {
    // Bounds in transients_ has already been recursed with (*this)(...)
    std::unordered_map<uint64_t, TransientBound> transients_;

    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    AnalyzeLinear analyzeLinear_;
    TypeInfer typeInfer_;
    GetHash getHash_;
    OutDatedBoundsRemover remover_;

  protected:
    CompTransientBounds() : typeInfer_(&buffers_), remover_(transients_) {}

    TransientBound transient(const Expr &op);
    DataType dtype(const Expr &op);
    uint64_t getHash(const Expr &op);

  private:
    static Expr sub1(const Expr &op);
    static Expr add1(const Expr &op);

    void applyCond(int k, const Expr &lhs, ASTNodeType opType, const Expr &rhs);
    void applyCond(const Expr &cond);

  protected:
    using Mutator::visit; // Avoid hiding virtual functions

    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Compute bounds of each UNIQUE (sub)expression
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
class CompUniqueBounds : public CompTransientBounds {
  public:
    typedef std::vector<LowerBound> LowerBoundsList;
    typedef std::vector<UpperBound> UpperBoundsList;
    typedef std::unordered_map<Expr, LowerBoundsList> LowerBoundsMap;
    typedef std::unordered_map<Expr, UpperBoundsList> UpperBoundsMap;

  private:
    LowerBoundsMap lower_;
    UpperBoundsMap upper_;

  protected:
    LowerBoundsList getLower(const Expr &op) const {
        return lower_.count(op) ? lower_.at(op) : LowerBoundsList{};
    }
    UpperBoundsList getUpper(const Expr &op) const {
        return upper_.count(op) ? upper_.at(op) : UpperBoundsList{};
    }

    template <class T> void setLower(const Expr &op, T &&list) {
        lower_[op] = std::forward<T>(list);
    }
    template <class T> void setUpper(const Expr &op, T &&list) {
        upper_[op] = std::forward<T>(list);
    }

    void updLower(LowerBoundsList &list, const LowerBound &bound) const;
    void updUpper(UpperBoundsList &list, const UpperBound &bound) const;

    int getIntLower(const Expr &op) const;
    int getIntUpper(const Expr &op) const;
    Ref<int> getInt(const Expr &op) const;

    bool alwaysLT(const Expr &lhs, const Expr &rhs) const;
    bool alwaysLE(const Expr &lhs, const Expr &rhs) const;

  public:
    const LowerBoundsMap &lower() const { return lower_; }
    const UpperBoundsMap &upper() const { return upper_; }

  protected:
    using CompTransientBounds::visit; // Avoid hiding virtual functions

    Expr visitExpr(const Expr &op,
                   const std::function<Expr(const Expr &)> &visitNode) override;

    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
    Expr visit(const IntConst &op) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
};

template <class BaseClass> class SimplifyPass : public BaseClass {
    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

    // Used to check for fixed point
    std::unordered_set<AST> mutated_;

    std::unordered_map<std::string, Expr> replace_;

  public:
    const std::unordered_set<AST> &mutated() const { return mutated_; }

  private:
    template <class T> T markMutated(const T &op) {
        auto ret = (*this)(op); // Recurse again to get bounds of op
        mutated_.insert(ret);
        return ret;
    }

  protected:
    using BaseClass::visit;

    Expr visitExpr(const Expr &op,
                   const std::function<Expr(const Expr &)> &visitNode) override;

    Expr visit(const Var &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const RoundTowards0Div &op) override;
    Expr visit(const Mod &op) override;
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
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

class BuiltinSimplify : public SimplifyPass<CompUniqueBounds> {};

class CheckFixedPoint : public Visitor {
  private:
    const std::unordered_set<AST> &mutated_;
    bool isFixPoint_ = true;

  public:
    CheckFixedPoint(const std::unordered_set<AST> &mutated)
        : mutated_(mutated) {}

    bool isFixPoint() const { return isFixPoint_; }

  protected:
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;
    void visitStmt(const Stmt &op,
                   const std::function<void(const Stmt &)> &visitNode) override;
};

/**
 * Simplify a program and compute bounds of each expressions
 *
 * This pass can only be applied on a complete program, instead of a single
 * expression, because it examines VarDef nodes of each Var
 *
 * @return : {simplified, lower, upper}
 */
template <class Simplifier>
std::tuple<Stmt, typename Simplifier::LowerBoundsMap,
           typename Simplifier::UpperBoundsMap>
simplifyAndGetBounds(const Stmt &op);

Stmt builtinSimplify(const Stmt &op);

Stmt simplifyPass(const Stmt &op);

} // namespace ir

#endif // SIMPLIFY_H
