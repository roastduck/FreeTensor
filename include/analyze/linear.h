#ifndef LINEAR_H
#define LINEAR_H

#include <unordered_map>
#include <unordered_set>

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
    GetHash getHash_;
    std::unordered_map<AST, LinearExpr> result_;
    std::unordered_set<AST> visited_;

  public:
    const std::unordered_map<AST, LinearExpr> &result() const {
        return result_;
    }

  protected:
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    // Note that for integer (floored) div, k * a / d !== (k / d) * a, so we are
    // not handling Div here
};

} // namespace ir

#endif // LINEAR_H
