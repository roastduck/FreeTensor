#ifndef FREE_TENSOR_GEN_PB_EXPR_H
#define FREE_TENSOR_GEN_PB_EXPR_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <hash.h>
#include <math/presburger.h>
#include <visitor.h>

namespace freetensor {

/**
 * Serialize expressions to an Presburger expression string
 *
 * If an expression or its sub-expression is not a Presburger expression, it
 * will be represented by a free variable. E.g., `a + b * (b + 1)` is
 * non-Presburger, because `b` is multiplied by a non-constant. This expression
 * will be converted to `a + free_var`. Since `b * (b + 1)` is already been
 * converted to `free_var`, the sub-expressions `b` and `b + 1` will be dropped,
 * although they are Presburger themselves. The free variable `free_var` will be
 * named by the expression itself, with an optional suffix.
 *
 * Sometimes there will be too many free varaibles if directly converted from a
 * user program. For example, if the program accesses an index `x + 2 * y`,
 * where `x` and `y` are unique to this access, this index can be simplify
 * represented by one free variable `z`, where `z = x + 2 * y`. If some
 * (sub-)expressions are not preferred to be a free variable, they can be
 * specified in the `noNeedToBeVars_` set.
 *
 * Use `GenPBExpr::gen` to generate a string, and its free variables
 */
class GenPBExpr : public Visitor {
  public:
    // hash -> presburger name
    typedef ASTHashMap<Expr, std::string> VarMap;

  private:
    std::unordered_map<Expr, std::string> results_;
    std::unordered_map<Expr, int>
        constants_; // isl detects contant literally, so we need to do constant
                    // folding here
    std::unordered_map<Expr, VarMap>
        vars_; // (sub-)expression -> free variables used inside
    Expr parent_ = nullptr;
    std::string varSuffix_;
    ASTHashSet<Expr> noNeedToBeVars_;

  public:
    GenPBExpr(const std::string &varSuffix = "",
              const ASTHashSet<Expr> &noNeedToBeVars = {})
        : varSuffix_(varSuffix), noNeedToBeVars_(noNeedToBeVars) {}

    const std::string &varSuffix() const { return varSuffix_; }

    std::pair<std::string, VarMap> gen(const Expr &op);

  protected:
    void visitExpr(const Expr &op) override;
    void visit(const Var &op) override;
    void visit(const IntConst &op) override;
    void visit(const BoolConst &op) override;
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
    void visit(const IfExpr &op) override;
    void visit(const Unbound &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_GEN_PB_EXPR_H
