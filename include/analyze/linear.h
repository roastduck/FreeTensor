#ifndef LINEAR_H
#define LINEAR_H

#include <unordered_map>

#include <analyze/hash.h>
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

} // namespace ir

#endif // LINEAR_H
