#ifndef ANALYZER_H
#define ANALYZER_H

#include <unordered_map>

#include <arith/hash.h>
#include <visitor.h>

namespace ir {

/**
 * k * a
 */
struct Scale {
    int k;
    Expr a;
};

/**
 * (sum_i k_i * a_i) + b
 */
struct LinearExpr {
    std::unordered_map<uint64_t, Scale> coeff_;
    int bias_;
};

/**
 * Try to represent each (sub)expression as a linear expression of memory
 * accesses and loop iterators
 */
class AnalyzeLinear : public Visitor {
    const std::unordered_map<const ExprNode *, uint64_t> &hash_; // expr -> hash
    std::unordered_map<const ASTNode *, LinearExpr> result_;     // hash -> expr

  public:
    AnalyzeLinear(const std::unordered_map<const ExprNode *, uint64_t> &hash)
        : hash_(hash) {}

    const std::unordered_map<const ASTNode *, LinearExpr> &result() const {
        return result_;
    }

  protected:
    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const Div &op) override;
};

/**
 * Try to get the upper bound and lower bound of each (sub)expression
 *
 * This pass is not accurate. Simplifying passes using thiss analysis may need
 * to run for multiple rounds
 */
class AnalyzeBounds : public Visitor {
    // expr -> hash
    const std::unordered_map<const ExprNode *, uint64_t> &hash_;
    const std::unordered_map<const ASTNode *, LinearExpr> &linear_;

    std::unordered_map<const ExprNode *, Expr> lower_, upper_;

    // buffer table
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    // iterator table
    std::unordered_map<std::string, std::pair<Expr, Expr>> iters_;

  private:
    // Compute k * a + b
    Expr compLinear(int k, const Expr &a, const Expr &b) const;

    // Optionally get lower bound
    Expr getLower(const LinearExpr &linear) const;
    Expr getUpper(const LinearExpr &linear) const;

    template <class T> void doAnalyze(const T &op) {
        if (linear_.count(op.get())) {
            auto &&lin = linear_.at(op.get());
            updLower(op, getLower(lin));
            updUpper(op, getUpper(lin));
        }
        Visitor::visit(op);
    }

    // Update lower bound, return old value
    void updLower(const Expr &op, const Expr &expr);
    void updUpper(const Expr &op, const Expr &expr);

    uint64_t getHash(const Expr &op);

  public:
    AnalyzeBounds(const std::unordered_map<const ExprNode *, uint64_t> &hash,
                  const std::unordered_map<const ASTNode *, LinearExpr> &linear)
        : hash_(hash), linear_(linear) {}

    const std::unordered_map<const ExprNode *, Expr> &lower() const {
        return lower_;
    }
    const std::unordered_map<const ExprNode *, Expr> &upper() const {
        return upper_;
    }

  protected:
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const Div &op) override;
    virtual void visit(const For &op) override;
};

} // namespace ir

#endif // ANALYZER_H
