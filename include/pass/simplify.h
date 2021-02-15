#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/bounds.h>
#include <analyze/linear.h>
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

/**
 * Try to get the upper bound and lower bound of each (sub)expression
 *
 * Inherit this pass to use it
 *
 * This pass is not accurate. Simplifying passes using this analysis may need
 * to run for multiple rounds
 */
class AnalyzeBounds : public Mutator {
  public:
    typedef std::unordered_map<Expr, std::vector<Bound>> BoundsMap;

  private:
    const std::unordered_map<Expr, uint64_t> &hash_; // expr -> hash

    BoundsMap lower_, upper_;

    // iterator table
    std::unordered_map<std::string, std::pair<Expr, Expr>> iters_;

  protected:
    std::vector<Bound> getLower(const Expr &op) const;
    std::vector<Bound> getUpper(const Expr &op) const;

    uint64_t getHash(const Expr &op);

  private:
    void updLower(const Expr &op, const Bound &bound);
    void updUpper(const Expr &op, const Bound &bound);

    int getIntLower(const Expr &op) const;
    int getIntUpper(const Expr &op) const;
    Ref<int> getInt(const Expr &op) const;

    static Expr sub1(const Expr &op);
    static Expr add1(const Expr &op);

  protected:
    AnalyzeBounds(const std::unordered_map<Expr, uint64_t> &hash)
        : hash_(hash) {}

  public:
    const BoundsMap &lower() const { return lower_; }
    const BoundsMap &upper() const { return upper_; }

  protected:
    using Mutator::visit; // Avoid hiding virtual functions

    virtual Expr visit(const Var &op) override;
    virtual Expr visit(const Load &op) override;
    virtual Expr visit(const IntConst &op) override;
    virtual Expr visit(const Add &op) override;
    virtual Expr visit(const Sub &op) override;
    virtual Expr visit(const Mul &op) override;
    virtual Expr visit(const Div &op) override;
    virtual Stmt visit(const For &op) override;
    virtual Stmt visit(const If &op) override;
};

class SimplifyPass : public AnalyzeBounds {
  public:
    typedef std::unordered_map<Expr, std::vector<Bound>> BoundsMap;

  private:
    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

    // Used to check for fixed point
    std::unordered_set<AST> mutated_;

  public:
    SimplifyPass(const std::unordered_map<Expr, uint64_t> &hash)
        : AnalyzeBounds(hash) {}

    const std::unordered_set<AST> &mutated() const { return mutated_; }

  private:
    template <class T> T markMutated(const T &op) {
        auto ret = (*this)(op); // Recurse again to get bounds of op
        mutated_.insert(ret);
        return ret;
    }

    template <class T> Expr doSimplify(const T &_op) {
        auto op = AnalyzeBounds::visit(_op);

        // To avoid divergence
        if (getHash(op) != getHash(_op)) {
            // E.g.
            // (1) a[0 - 0] -> a[0]
            // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
            return op;
        }

        Expr best = nullptr;
        auto bestScope = -1;
        for (auto &&lower : getLower(op)) {
            auto hl = getHash(lower.expr_);
            for (auto &&upper : getUpper(op)) {
                auto hr = getHash(upper.expr_);
                if (hl == hr) {
                    // We need to choose the simplest one. Other wise
                    // we are always picking the original expression
                    auto scope = findInnerMostScope(varScope_, lower.expr_);
                    if (!best.isValid() || scope < bestScope) {
                        best = lower.expr_, bestScope = scope;
                    }
                    break;
                }
            }
        }
        if (best.isValid() && getHash(best) != getHash(op)) {
            return markMutated(best);
        }
        return op;
    }

    bool checkUpperCmp0(const Expr &normForm,
                        const std::function<bool(int, int)> &&cmp);
    bool checkLowerCmp0(const Expr &normForm,
                        const std::function<bool(int, int)> &&cmp);

  protected:
    using AnalyzeBounds::visit;

    Expr visit(const Var &op) override { return doSimplify(op); }
    Expr visit(const Add &op) override { return doSimplify(op); }
    Expr visit(const Sub &op) override { return doSimplify(op); }
    Expr visit(const Mul &op) override { return doSimplify(op); }
    Expr visit(const Div &op) override;
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

Stmt simplifyPass(const Stmt &op);

// return {simplified, lower, upper}
std::tuple<Stmt, SimplifyPass::BoundsMap, SimplifyPass::BoundsMap>
simplifyAndGetBounds(const Stmt &op);

} // namespace ir

#endif // SIMPLIFY_H
