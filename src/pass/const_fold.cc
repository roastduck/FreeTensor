#include <math/utils.h>
#include <pass/const_fold.h>

namespace freetensor {

#define BINARY_OP(OPNAME, OP)                                                  \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
            requires requires(T l, U r) { l OP r; }                            \
        auto operator()(const T &l, const U &r) {                              \
            return l OP r;                                                     \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r) -> decltype(l) {               \
            ERROR("Invalid operator " #OPNAME " on given types");              \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r); },          \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define BINARY_OP_F(OPNAME, OP_F)                                              \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
            requires requires(T l, U r) { (OP_F)(l, r); }                      \
        auto operator()(const T &l, const U &r) {                              \
            return (OP_F)(l, r);                                               \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r) -> decltype(l) {               \
            ERROR("Invalid operator " #OPNAME " on given types");              \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r); },          \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define UNARY_OP(OPNAME, OP)                                                   \
    struct op_f_##OPNAME {                                                     \
        template <typename T>                                                  \
            requires requires(T t) { OP(t); }                                  \
        auto operator()(const T &t) {                                          \
            return OP(t);                                                      \
        }                                                                      \
        template <typename T> auto operator()(const T &t) -> decltype(t) {     \
            ERROR("Invalid operator " #OPNAME " on given types");              \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitUnary(                                                     \
            op, [](auto x) { return op_f_##OPNAME()(x); },                     \
            [](auto x) { return make##OPNAME(x); });                           \
    }

BINARY_OP(Add, +)
BINARY_OP(Sub, -)
BINARY_OP(Mul, *)
BINARY_OP_F(RealDiv, realDiv)
BINARY_OP_F(FloorDiv, floorDiv)
BINARY_OP_F(CeilDiv, ceilDiv)
BINARY_OP(RoundTowards0Div, /)
BINARY_OP_F(Mod, mod)
BINARY_OP(Remainder, %)
BINARY_OP_F(Min, [](auto &&lhs, auto &&rhs) {
    typedef decltype(lhs + rhs) V;
    return std::min<V>(lhs, rhs);
})
BINARY_OP_F(Max, [](auto &&lhs, auto &&rhs) {
    typedef decltype(lhs + rhs) V;
    return std::max<V>(lhs, rhs);
})
BINARY_OP(LT, <)
BINARY_OP(LE, <=)
BINARY_OP(GT, >)
BINARY_OP(GE, >=)
BINARY_OP(EQ, ==)
BINARY_OP(NE, !=)
BINARY_OP(LAnd, &&)
BINARY_OP(LOr, ||)
UNARY_OP(LNot, !)
UNARY_OP(Sqrt, std::sqrt)
UNARY_OP(Exp, std::exp)
UNARY_OP(Square, square)
UNARY_OP(Sigmoid, sigmoid)
UNARY_OP(Sin, std::sin)
UNARY_OP(Cos, std::cos)
UNARY_OP(Tan, std::tan)
UNARY_OP(Tanh, std::tanh)

// Avoid -Wabsolute-value in Clang
static int64_t _abs(const int64_t &t) { return std::abs(t); }
static double _abs(const double &t) { return std::abs(t); }

UNARY_OP(Abs, _abs)
UNARY_OP(Floor, std::floor)
UNARY_OP(Ceil, std::ceil)

Expr ConstFold::visit(const Unbound &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Unbound);
    auto op = __op.as<UnboundNode>();
    if (op->expr_->isConst()) {
        return op->expr_;
    }
    return op;
}

Expr ConstFold::visit(const Cast &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Cast);
    auto op = __op.as<CastNode>();
    if (op->expr_->isConst() && (op->destType_ != DataType::Custom)) {
        return castType(op->destType_, op->expr_.as<ConstNode>());
    }
    if (op->expr_->dtype() == op->destType_) {
        return op->expr_; // FIXME: This may break assertions if we inherit
                          // ConstFold in other passes
    }
    return op;
}

Expr ConstFold::visit(const IfExpr &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.as<IfExprNode>();
    if (op->cond_->isConst() && op->thenCase_->isConst() &&
        op->elseCase_->isConst()) {
        // We only handle purely constant cases in const_fold, so that other
        // Mutators that inherits ConstFold receives either a constant, or the
        // original node
        ASSERT(op->cond_->nodeType() == ASTNodeType::BoolConst);
        return op->cond_.as<BoolConstNode>()->val_ ? op->thenCase_
                                                   : op->elseCase_;
    }
    return op;
}

} // namespace freetensor
