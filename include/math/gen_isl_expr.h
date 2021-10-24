#ifndef GEN_ISL_EXPR_H
#define GEN_ISL_EXPR_H

#include <unordered_map>
#include <unordered_set>

#include <math/isl.h>
#include <visitor.h>

namespace ir {

/**
 * Serialize expressions to an ISL input string
 *
 * It returns nullptr for unsupported expressions, because ISL reports errors on
 * them
 */
class GenISLExpr : public Visitor {
  protected:
    std::unordered_map<Expr, std::string> results_;
    std::unordered_set<Expr> visited_;
    std::unordered_map<Expr, int> constants_;
    std::unordered_map<std::string, std::string> idCache_; // IR IDs -> ISL IDs
    std::unordered_set<std::string> idFlag_;               // ISL IDs

  public:
    std::string normalizeId(const std::string &id);

    Ref<std::string> gen(const Expr &op);

  protected:
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;
    void visit(const Var &op) override;
    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const LAnd &op) override;
    void visit(const LOr &op) override;
    void visit(const LNot &op) override;
    void visit(const LT &op) override;
    void visit(const LE &op) override;
    void visit(const GT &op) override;
    void visit(const GE &op) override;
    void visit(const EQ &op) override;
    void visit(const NE &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const Mod &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
};

} // namespace ir

#endif // GEN_ISL_EXPR_H
