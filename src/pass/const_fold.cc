#include <math/utils.h>
#include <pass/const_fold.h>

namespace ir {

#define BINARY_OP(OPNAME, OP)                                                  \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, int) -> decltype(l OP r) {     \
            return l OP r;                                                     \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, char) -> decltype(l) {         \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return l;                                                          \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r, 0); },       \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define BINARY_OP_F(OPNAME, OP_F, OP_TYPE_HINT)                                \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, int)                           \
            -> decltype(l OP_TYPE_HINT r) {                                    \
            typedef decltype(l OP_TYPE_HINT r) V;                              \
            return (OP_F)((V)l, (V)r);                                         \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, char) -> decltype(l) {         \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return l;                                                          \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r, 0); },       \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define UNARY_OP(OPNAME, OP)                                                   \
    struct op_f_##OPNAME {                                                     \
        template <typename T>                                                  \
        auto operator()(const T &t, int) -> decltype(OP(t)) {                  \
            return OP(t);                                                      \
        }                                                                      \
        template <typename T>                                                  \
        auto operator()(const T &t, char) -> decltype(t) {                     \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return t;                                                          \
        }                                                                      \
    };                                                                         \
    Expr ConstFold::visit(const OPNAME &op) {                                  \
        return visitUnary(                                                     \
            op, [](auto x) { return op_f_##OPNAME()(x, 0); },                  \
            [](auto x) { return make##OPNAME(x); });                           \
    }

BINARY_OP(Add, +)
BINARY_OP(Sub, -)
BINARY_OP(Mul, *)
BINARY_OP_F(RealDiv, realDiv, /)
BINARY_OP_F(FloorDiv, floorDiv, %)
BINARY_OP_F(CeilDiv, ceilDiv, %)
BINARY_OP(RoundTowards0Div, /)
BINARY_OP_F(Mod, mod, %)
BINARY_OP(Remainder, %)
BINARY_OP_F(Min, std::min, +)
BINARY_OP_F(Max, std::max, +)
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
UNARY_OP(Tanh, std::tanh)

// Avoid -Wabsolute-value in Clang
static int64_t _abs(const int64_t &t) { return std::abs(t); }
static double _abs(const double &t) { return std::abs(t); }

UNARY_OP(Abs, _abs)
UNARY_OP(Floor, std::floor)
UNARY_OP(Ceil, std::ceil)

Expr ConstFold::visit(const Cast &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Cast);
    auto op = __op.as<CastNode>();
    if (op->expr_->isConst() &&
        (op->dtype_ == DataType::Bool || op->dtype_ == DataType::Float32 ||
         op->dtype_ == DataType::Float64 || op->dtype_ == DataType::Int64 ||
         op->dtype_ == DataType::Int32)) {
        return castType(op->dtype_, op->expr_.as<ConstNode>());
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

} // namespace ir
