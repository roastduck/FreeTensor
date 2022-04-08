#ifndef EXPR_H
#define EXPR_H

#include <string>
#include <vector>

#include <ast.h>
#include <data_type.h>

namespace ir {

/**
 * Matches any expression
 *
 * Only used in pattern matching
 */
class AnyExprNode : public ExprNode {
  public:
    void compHash() override;
    DEFINE_NODE_TRAIT(AnyExpr);
};
typedef Ref<AnyExprNode> AnyExpr;
#define makeAnyExpr(...) makeNode(AnyExpr, __VA_ARGS__)
inline Expr _makeAnyExpr() { return AnyExpr::make(); }

class VarNode : public ExprNode {
  public:
    std::string name_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Var);
};
typedef Ref<VarNode> Var;
#define makeVar(...) makeNode(Var, __VA_ARGS__)
inline Expr _makeVar(const std::string &name) {
    Var v = Var::make();
    v->name_ = name;
    return v;
}

class LoadNode : public ExprNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_ = ChildOf{this};
    void compHash() override;
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
#define makeLoad(...) makeNode(Load, __VA_ARGS__)
template <class Tindices>
Expr _makeLoad(const std::string &var, Tindices &&indices) {
    Load l = Load::make();
    l->var_ = var;
    l->indices_ = std::forward<Tindices>(indices);
    return l;
}
inline Expr _makeLoad(const std::string &var,
                      const std::vector<Expr> &indices) {
    Load l = Load::make();
    l->var_ = var;
    l->indices_ = indices;
    return l;
}

class ConstNode : public ExprNode {
  public:
    bool isConst() const override { return true; }
};
typedef Ref<ConstNode> Const;

class IntConstNode : public ConstNode {
  public:
    int64_t val_;
    void compHash() override;
    DEFINE_NODE_TRAIT(IntConst);
};
typedef Ref<IntConstNode> IntConst;
#define makeIntConst(...) makeNode(IntConst, __VA_ARGS__)
inline Expr _makeIntConst(int64_t val) {
    IntConst c = IntConst::make();
    c->val_ = val;
    return c;
}

class FloatConstNode : public ConstNode {
  public:
    double val_;
    void compHash() override;
    DEFINE_NODE_TRAIT(FloatConst);
};
typedef Ref<FloatConstNode> FloatConst;
#define makeFloatConst(...) makeNode(FloatConst, __VA_ARGS__)
inline Expr _makeFloatConst(double val) {
    FloatConst c = FloatConst::make();
    c->val_ = val;
    return c;
}

class BoolConstNode : public ConstNode {
  public:
    bool val_;
    void compHash() override;
    DEFINE_NODE_TRAIT(BoolConst);
};
typedef Ref<BoolConstNode> BoolConst;
#define makeBoolConst(...) makeNode(BoolConst, __VA_ARGS__)
inline Expr _makeBoolConst(bool val) {
    BoolConst b = BoolConst::make();
    b->val_ = val;
    return b;
}

class BinaryExprNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_ = ChildOf{this}, rhs_ = ChildOf{this};

    bool isBinary() const override { return true; }
    virtual bool isCommutative() const = 0;
};
typedef Ref<BinaryExprNode> BinaryExpr;

class CommutativeBinaryExprNode : public BinaryExprNode {
  public:
    void compHash() override;
    bool isCommutative() const override { return true; }
};
typedef Ref<CommutativeBinaryExprNode> CommutativeBinaryExpr;

class NonCommutativeBinaryExprNode : public BinaryExprNode {
  public:
    void compHash() override;
    bool isCommutative() const override { return false; }
};
typedef Ref<NonCommutativeBinaryExprNode> NonCommutativeBinaryExpr;

class AddNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Add);
};
typedef Ref<AddNode> Add;
#define makeAdd(...) makeNode(Add, __VA_ARGS__)
template <class T, class U> Expr _makeAdd(T &&lhs, U &&rhs) {
    Add a = Add::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class SubNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Sub);
};
typedef Ref<SubNode> Sub;
#define makeSub(...) makeNode(Sub, __VA_ARGS__)
template <class T, class U> Expr _makeSub(T &&lhs, U &&rhs) {
    Sub a = Sub::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MulNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Mul);
};
typedef Ref<MulNode> Mul;
#define makeMul(...) makeNode(Mul, __VA_ARGS__)
template <class T, class U> Expr _makeMul(T &&lhs, U &&rhs) {
    Mul a = Mul::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/**
 * Floating-point division
 */
class RealDivNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(RealDiv);
};
typedef Ref<RealDivNode> RealDiv;
#define makeRealDiv(...) makeNode(RealDiv, __VA_ARGS__)
template <class T, class U> Expr _makeRealDiv(T &&lhs, U &&rhs) {
    RealDiv a = RealDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/**
 * Integer division, rounded towards negative infinity
 *
 * FloorDiv nodes are easy to analyze, and will be replaced by RoundTowards0Div
 * nodes before codegen if possible
 */
class FloorDivNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(FloorDiv);
};
typedef Ref<FloorDivNode> FloorDiv;
#define makeFloorDiv(...) makeNode(FloorDiv, __VA_ARGS__)
template <class T, class U> Expr _makeFloorDiv(T &&lhs, U &&rhs) {
    FloorDiv a = FloorDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/**
 * Integer division, rounded towards positive infinity
 *
 * CeilDiv nodes are easy to analyze, and will be replaced by RoundTowards0Div
 * nodes before codegen if possible
 */
class CeilDivNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(CeilDiv);
};
typedef Ref<CeilDivNode> CeilDiv;
#define makeCeilDiv(...) makeNode(CeilDiv, __VA_ARGS__)
template <class T, class U> Expr _makeCeilDiv(T &&lhs, U &&rhs) {
    CeilDiv a = CeilDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/**
 * Integer division, rounded towards 0
 *
 * RoundTowards0Div nodes comply with the integer division behaviour in C. They
 * have minimal runtime overhead, but are hard to analyze
 */
class RoundTowards0DivNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(RoundTowards0Div);
};
typedef Ref<RoundTowards0DivNode> RoundTowards0Div;
#define makeRoundTowards0Div(...) makeNode(RoundTowards0Div, __VA_ARGS__)
template <class T, class U> Expr _makeRoundTowards0Div(T &&lhs, U &&rhs) {
    RoundTowards0Div a = RoundTowards0Div::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/** Modulo
 *
 * Mod(3, 5) = 3
 * Mod(-3, 5) = 2
 */
class ModNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Mod);
};
typedef Ref<ModNode> Mod;
#define makeMod(...) makeNode(Mod, __VA_ARGS__)
template <class T, class U> Expr _makeMod(T &&lhs, U &&rhs) {
    Mod a = Mod::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/** Remainder
 *
 * Remainder(3, 5) = 3
 * Remainder(-3, 5) = -3
 */
class RemainderNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Remainder);
};
typedef Ref<RemainderNode> Remainder;
#define makeRemainder(...) makeNode(Remainder, __VA_ARGS__)
template <class T, class U> Expr _makeRemainder(T &&lhs, U &&rhs) {
    Remainder a = Remainder::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MinNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Min);
};
typedef Ref<MinNode> Min;
#define makeMin(...) makeNode(Min, __VA_ARGS__)
template <class T, class U> Expr _makeMin(T &&lhs, U &&rhs) {
    Min m = Min::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class MaxNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(Max);
};
typedef Ref<MaxNode> Max;
#define makeMax(...) makeNode(Max, __VA_ARGS__)
template <class T, class U> Expr _makeMax(T &&lhs, U &&rhs) {
    Max m = Max::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class LTNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(LT);
};
typedef Ref<LTNode> LT;
#define makeLT(...) makeNode(LT, __VA_ARGS__)
template <class T, class U> Expr _makeLT(T &&lhs, U &&rhs) {
    LT a = LT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LENode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(LE);
};
typedef Ref<LENode> LE;
#define makeLE(...) makeNode(LE, __VA_ARGS__)
template <class T, class U> Expr _makeLE(T &&lhs, U &&rhs) {
    LE a = LE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GTNode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(GT);
};
typedef Ref<GTNode> GT;
#define makeGT(...) makeNode(GT, __VA_ARGS__)
template <class T, class U> Expr _makeGT(T &&lhs, U &&rhs) {
    GT a = GT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GENode : public NonCommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(GE);
};
typedef Ref<GENode> GE;
#define makeGE(...) makeNode(GE, __VA_ARGS__)
template <class T, class U> Expr _makeGE(T &&lhs, U &&rhs) {
    GE a = GE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class EQNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(EQ);
};
typedef Ref<EQNode> EQ;
#define makeEQ(...) makeNode(EQ, __VA_ARGS__)
template <class T, class U> Expr _makeEQ(T &&lhs, U &&rhs) {
    EQ a = EQ::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class NENode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
#define makeNE(...) makeNode(NE, __VA_ARGS__)
template <class T, class U> Expr _makeNE(T &&lhs, U &&rhs) {
    NE a = NE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LAndNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(LAnd);
};
typedef Ref<LAndNode> LAnd;
#define makeLAnd(...) makeNode(LAnd, __VA_ARGS__)
template <class T, class U> Expr _makeLAnd(T &&lhs, U &&rhs) {
    LAnd l = LAnd::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class LOrNode : public CommutativeBinaryExprNode {
    DEFINE_NODE_TRAIT(LOr);
};
typedef Ref<LOrNode> LOr;
#define makeLOr(...) makeNode(LOr, __VA_ARGS__)
template <class T, class U> Expr _makeLOr(T &&lhs, U &&rhs) {
    LOr l = LOr::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class UnaryExprNode : public ExprNode {
  public:
    SubTree<ExprNode> expr_ = ChildOf{this};

    void compHash() override;
    bool isUnary() const override { return true; }
};
typedef Ref<UnaryExprNode> UnaryExpr;

class LNotNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(LNot);
};
typedef Ref<LNotNode> LNot;
#define makeLNot(...) makeNode(LNot, __VA_ARGS__)
template <class T> Expr _makeLNot(T &&expr) {
    LNot n = LNot::make();
    n->expr_ = std::forward<T>(expr);
    return n;
}

class SqrtNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Sqrt);
};
typedef Ref<SqrtNode> Sqrt;
#define makeSqrt(...) makeNode(Sqrt, __VA_ARGS__)
template <class T> Expr _makeSqrt(T &&expr) {
    Sqrt s = Sqrt::make();
    s->expr_ = std::forward<T>(expr);
    return s;
}

class ExpNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Exp);
};
typedef Ref<ExpNode> Exp;
#define makeExp(...) makeNode(Exp, __VA_ARGS__)
template <class T> Expr _makeExp(T &&expr) {
    Exp e = Exp::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class SquareNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Square);
};
typedef Ref<SquareNode> Square;
#define makeSquare(...) makeNode(Square, __VA_ARGS__)
template <class T> Expr _makeSquare(T &&expr) {
    Square e = Square::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class SigmoidNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Sigmoid);
};
typedef Ref<SigmoidNode> Sigmoid;
#define makeSigmoid(...) makeNode(Sigmoid, __VA_ARGS__)
template <class T> Expr _makeSigmoid(T &&expr) {
    Sigmoid e = Sigmoid::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class TanhNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Tanh);
};
typedef Ref<TanhNode> Tanh;
#define makeTanh(...) makeNode(Tanh, __VA_ARGS__)
template <class T> Expr _makeTanh(T &&expr) {
    Tanh e = Tanh::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class AbsNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Abs);
};
typedef Ref<AbsNode> Abs;
#define makeAbs(...) makeNode(Abs, __VA_ARGS__)
template <class T> Expr _makeAbs(T &&expr) {
    Abs e = Abs::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class FloorNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Floor);
};
typedef Ref<FloorNode> Floor;
#define makeFloor(...) makeNode(Floor, __VA_ARGS__)
template <class T> Expr _makeFloor(T &&expr) {
    Floor e = Floor::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class CeilNode : public UnaryExprNode {
    DEFINE_NODE_TRAIT(Ceil);
};
typedef Ref<CeilNode> Ceil;
#define makeCeil(...) makeNode(Ceil, __VA_ARGS__)
template <class T> Expr _makeCeil(T &&expr) {
    Ceil e = Ceil::make();
    e->expr_ = std::forward<T>(expr);
    return e;
}

class IfExprNode : public ExprNode {
  public:
    SubTree<ExprNode> cond_ = ChildOf{this};
    SubTree<ExprNode> thenCase_ = ChildOf{this};
    SubTree<ExprNode> elseCase_ = ChildOf{this};
    void compHash() override;
    DEFINE_NODE_TRAIT(IfExpr);
};
typedef Ref<IfExprNode> IfExpr;
#define makeIfExpr(...) makeNode(IfExpr, __VA_ARGS__)
template <class T, class U, class V>
Expr _makeIfExpr(T &&cond, U &&thenCase, V &&elseCase) {
    IfExpr e = IfExpr::make();
    e->cond_ = std::forward<T>(cond);
    e->thenCase_ = std::forward<U>(thenCase);
    e->elseCase_ = std::forward<V>(elseCase);
    return e;
}

class CastNode : public ExprNode {
  public:
    SubTree<ExprNode> expr_ = ChildOf{this};
    DataType dtype_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Cast);
};
typedef Ref<CastNode> Cast;
#define makeCast(...) makeNode(Cast, __VA_ARGS__)
template <class T> Expr _makeCast(T &&expr, DataType dtype) {
    Cast e = Cast::make();
    e->expr_ = std::forward<T>(expr);
    e->dtype_ = dtype;
    return e;
}

/**
 * Invoke whatever target code
 */
class IntrinsicNode : public ExprNode {
  public:
    std::string format_; /// what to run. "%" is filled by parameters one by one
                         /// E.g. sinf(%)
    SubTreeList<ExprNode> params_ = ChildOf{this};
    DataType retType_;
    bool hasSideEffect_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Intrinsic);
};
typedef Ref<IntrinsicNode> Intrinsic;
#define makeIntrinsic(...) makeNode(Intrinsic, __VA_ARGS__)
template <class T>
Expr _makeIntrinsic(const std::string &format, T &&params, DataType retType,
                    bool hasSideEffect) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = std::forward<T>(params);
    i->retType_ = retType;
    i->hasSideEffect_ = hasSideEffect;
    return i;
}
inline Expr _makeIntrinsic(const std::string &format,
                           std::initializer_list<Expr> params, DataType retType,
                           bool hasSideEffect) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = params;
    i->retType_ = retType;
    i->hasSideEffect_ = hasSideEffect;
    return i;
}

template <class T, class U>
Expr makeBinary(ASTNodeType nodeType, T &&lhs, U &&rhs) {
    switch (nodeType) {
    case ASTNodeType::Add:
        return makeAdd(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Sub:
        return makeSub(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Mul:
        return makeMul(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::RealDiv:
        return makeRealDiv(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::FloorDiv:
        return makeFloorDiv(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::CeilDiv:
        return makeCeilDiv(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::RoundTowards0Div:
        return makeRoundTowards0Div(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Mod:
        return makeMod(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Remainder:
        return makeRemainder(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Min:
        return makeMin(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::Max:
        return makeMax(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::LT:
        return makeLT(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::LE:
        return makeLE(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::GT:
        return makeGT(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::GE:
        return makeGE(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::EQ:
        return makeEQ(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::NE:
        return makeNE(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::LAnd:
        return makeLAnd(std::forward<T>(lhs), std::forward<U>(rhs));
    case ASTNodeType::LOr:
        return makeLOr(std::forward<T>(lhs), std::forward<U>(rhs));
    default:
        ASSERT(false);
    }
}

template <class T> Expr makeUnary(ASTNodeType nodeType, T &&expr) {
    switch (nodeType) {
    case ASTNodeType::LNot:
        return makeLNot(std::forward<T>(expr));
    case ASTNodeType::Sqrt:
        return makeSqrt(std::forward<T>(expr));
    case ASTNodeType::Exp:
        return makeExp(std::forward<T>(expr));
    case ASTNodeType::Square:
        return makeSquare(std::forward<T>(expr));
    case ASTNodeType::Sigmoid:
        return makeSigmoid(std::forward<T>(expr));
    case ASTNodeType::Tanh:
        return makeTanh(std::forward<T>(expr));
    default:
        ASSERT(false);
    }
}

} // namespace ir

#endif // EXPR_H
