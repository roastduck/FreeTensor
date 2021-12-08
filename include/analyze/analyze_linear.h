#ifndef ANALYZE_LINEAR_H
#define ANALYZE_LINEAR_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <math/linear.h>
#include <visitor.h>

namespace ir {

/**
 * Try to represent each (sub)expression as a linear expression of memory
 * accesses and loop iterators
 */
class AnalyzeLinear : public Visitor {
    GetHash getHash_;
    std::unordered_map<AST, LinearExpr<int64_t>> result_;

  public:
    const std::unordered_map<AST, LinearExpr<int64_t>> &result() const {
        return result_;
    }

  protected:
    void visitExpr(const Expr &op) override;

    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    // Note that for integer (floored) div, k * a / d !== (k / d) * a, so we are
    // not handling Div here
};

} // namespace ir

#endif // ANALYZE_LINEAR_H
