#ifndef FREE_TENSOR_FLOAT_SIMPLIFY_H
#define FREE_TENSOR_FLOAT_SIMPLIFY_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>
#include <pass/const_fold.h>

namespace freetensor {

class FloatSimplify : public SymbolTable<ConstFold> {
    typedef SymbolTable<ConstFold> BaseClass;

    template <class T> bool equals(const Expr &op, T &&val) const {
        if (op->nodeType() == ASTNodeType::IntConst &&
            op.as<IntConstNode>()->val_ == val) {
            return true;
        }
        if (op->nodeType() == ASTNodeType::FloatConst &&
            op.as<FloatConstNode>()->val_ == val) {
            return true;
        }
        return false;
    }

    Expr normalizeRealMulDiv(const Expr &op);

  protected:
    using BaseClass::visit;
    Expr visitExpr(const Expr &expr) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const RealDiv &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
    Expr visit(const Sqrt &op) override;
    Expr visit(const Square &op) override;
    Expr visit(const Abs &op) override;
    // TODO: Cancel Exp against Ln
};

/**
 * Simplify floating-point expressions in an AST
 */
Stmt floatSimplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(floatSimplify)

} // namespace freetensor

#endif // FREE_TENSOR_FLOAT_SIMPLIFY_H
