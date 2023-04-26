#ifndef FREE_TENSOR_CONST_FOLD_H
#define FREE_TENSOR_CONST_FOLD_H

#include <func.h>
#include <mutator.h>

namespace freetensor {

/**
 * Calculate constant (sub)expressions
 *
 * For propagating constants among statements, please refer to
 * pass/scalar_prop_const and pass/tensor_prop_const
 *
 * This Mutator can be inherited
 */
class ConstFold : public Mutator {
    /**
     * @brief Type dispatch for constant types.
     *
     * @tparam F Functor type for given callback
     * @param c Reference of the constant AST node
     * @param f Callback for processing a typed constant value, should accept
     * any concrete type (through an `auto`/templated parameter)
     * @return auto Returns what `f` returns.
     */
    template <typename F> static auto dispatch(const Const &c, F f) {
        switch (c->nodeType()) {
        case ASTNodeType::IntConst:
            return f(c.as<IntConstNode>()->val_);
        case ASTNodeType::FloatConst:
            return f(c.as<FloatConstNode>()->val_);
        case ASTNodeType::BoolConst:
            return f(c.as<BoolConstNode>()->val_);
        default:
            ASSERT(false && "Unknown Const node");
        }
    }

    /**
     * @brief Wrap a typed value into a constant AST node.
     *
     * @param t Compile-time values to be wrapped
     * @return Const Wrapped `Const` AST node
     * @{
     */
    static Const wrap(const int &t) { return makeIntConst(t).as<ConstNode>(); }

    static Const wrap(const int64_t &t) {
        return makeIntConst(t).as<ConstNode>();
    }

    static Const wrap(const double &t) {
        return makeFloatConst(t).as<ConstNode>();
    }

    static Const wrap(const bool &t) {
        return makeBoolConst(t).as<ConstNode>();
    }
    /** @} */

    /**
     * @brief Generic binary operation visit
     *
     * @tparam F Functor type of the callback
     * @tparam FAlt Functor type for recovering the node
     * @param op The BinaryExpr to visit
     * @param f Callback for constant folding over two statically typed values
     * @param falt Callback for recovering the node
     * @return Expr Result expression, possibly constant-folded
     */
    template <typename F, typename FAlt>
    Expr visitBinary(const BinaryExpr &op, F f, FAlt falt) {
        auto lhs = (*this)(op->lhs_);
        auto rhs = (*this)(op->rhs_);
        if (lhs->isConst() && rhs->isConst()) {
            return dispatch(lhs.as<ConstNode>(), [&](auto ll) {
                return dispatch(rhs.as<ConstNode>(),
                                [&](auto rr) { return wrap(f(ll, rr)); });
            });
        } else {
            return falt(lhs, rhs);
        }
    }

    /**
     * @brief Generic unary operation visit
     *
     * @tparam F Functor type of the callback
     * @tparam FAlt Functor type for recovering the node
     * @param op The UnaryExpr to visit
     * @param f Callback for constant folding over a statically typed value
     * @param falt Callback for recovering the node
     * @return Expr Result expression, possibly constant-folded
     */
    template <typename F, typename FAlt>
    Expr visitUnary(const UnaryExpr &op, F f, FAlt falt) {
        auto x = (*this)(op->expr_);
        if (x->isConst()) {
            return dispatch(x.as<ConstNode>(),
                            [&](auto xx) { return wrap(f(xx)); });
        } else {
            return falt(x);
        }
    }

  protected:
    /**
     * @brief Cast the data type of a `Const` node.
     *
     * @param type Target type
     * @param val Constant node to be casted
     * @return Const Casted constant node
     */
    static Const castType(DataType type, const Const &val) {
        return dispatch(val, [type](auto v) {
            switch (type.base()) {
            case DataType::Int32:
            case DataType::Int64:
                return wrap(int64_t(v));
            case DataType::Float32:
            case DataType::Float64:
                return wrap(double(v));
            case DataType::Bool:
                return wrap(bool(v));
            default:
                ASSERT(false && "Unrecognized variable type assigned");
            }
        });
    }

  protected:
    using Mutator::visit;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const RealDiv &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const RoundTowards0Div &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Remainder &op) override;
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
    Expr visit(const Sqrt &op) override;
    Expr visit(const Exp &op) override;
    Expr visit(const Square &op) override;
    Expr visit(const Sigmoid &op) override;
    Expr visit(const Sin &op) override;
    Expr visit(const Cos &op) override;
    Expr visit(const Tan &op) override;
    Expr visit(const Tanh &op) override;
    Expr visit(const Abs &op) override;
    Expr visit(const Floor &op) override;
    Expr visit(const Ceil &op) override;
    Expr visit(const Cast &op) override;
    Expr visit(const IfExpr &op) override;
};

/**
 * Calculate constant (sub)expressions
 *
 * For propagating constants among statements, please refer to
 * pass/scalar_prop_const and pass/tensor_prop_const
 */
inline Stmt constFold(const Stmt &op) { return ConstFold()(op); }
inline Expr constFold(const Expr &op) { return ConstFold()(op); }

DEFINE_PASS_FOR_FUNC(constFold)

} // namespace freetensor

#endif // FREE_TENSOR_CONST_FOLD_H
