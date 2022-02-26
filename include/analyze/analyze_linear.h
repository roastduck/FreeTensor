#ifndef ANALYZE_LINEAR_H
#define ANALYZE_LINEAR_H

#include <unordered_map>
#include <unordered_set>

#include <hash.h>
#include <math/linear.h>
#include <opt.h>
#include <visitor.h>

namespace ir {

class AnalyzeLinear : public Visitor {
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

/**
 * Try to represent each (sub)expression as a linear expression of memory
 * accesses and loop iterators
 */
LinearExpr<int64_t> linear(const Expr &expr);

/**
 * Try to represent an comparison as a "LINEAR OP 0" form
 */
Opt<std::pair<LinearExpr<int64_t>, ASTNodeType>> linearComp(const Expr &expr);

} // namespace ir

#endif // ANALYZE_LINEAR_H
