#ifndef FREE_TENSOR_EXPR_H
#define FREE_TENSOR_EXPR_H

#include <string>
#include <vector>

#include <ast.h>
#include <type/data_type.h>

namespace freetensor {

/**
 * Any expression
 *
 * Only used in pattern matching and type inference
 */
class AnyExprNode : public ExprNode {
  public:
    void compHash() override;
    void inferDType() override { ASSERT(false); }
    std::vector<Expr> children() const override { return {}; }
    DEFINE_NODE_TRAIT(AnyExpr);
};
typedef Ref<AnyExprNode> AnyExpr;
inline Expr
makeAnyExpr(std::source_location loc = std::source_location::current()) {
    AnyExpr a = AnyExpr::make();
    a->setDebugBlame(loc);
    return a;
}

class VarNode : public ExprNode {
  public:
    std::string name_;
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override { return {}; }
    DEFINE_NODE_TRAIT(Var);
};
typedef Ref<VarNode> Var;
inline Expr
makeVar(const std::string &name,
        std::source_location loc = std::source_location::current()) {
    ASSERT(!name.empty());
    Var v = Var::make();
    v->name_ = name;
    v->setDebugBlame(loc);
    return v;
}

class LoadNode : public ExprNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_ = ChildOf{this};
    DataType loadType_;
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override { return indices_; }
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
template <class Tindices>
Expr makeLoad(const std::string &var, Tindices &&indices, DataType loadType,
              std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Load l = Load::make();
    l->var_ = var;
    l->indices_ = std::forward<Tindices>(indices);
    l->loadType_ = loadType;
    l->setDebugBlame(loc);
    return l;
}
inline Expr
makeLoad(const std::string &var, const std::vector<Expr> &indices,
         DataType loadType,
         std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Load l = Load::make();
    l->var_ = var;
    l->indices_ = indices;
    l->loadType_ = loadType;
    l->setDebugBlame(loc);
    return l;
}

class ConstNode : public ExprNode {
  public:
    bool isConst() const override { return true; }
    std::vector<Expr> children() const override { return {}; }
};
typedef Ref<ConstNode> Const;

class IntConstNode : public ConstNode {
  public:
    int64_t val_;
    void compHash() override;
    void inferDType() override;
    DEFINE_NODE_TRAIT(IntConst);
};
typedef Ref<IntConstNode> IntConst;
inline Expr
makeIntConst(int64_t val,
             std::source_location loc = std::source_location::current()) {
    IntConst c = IntConst::make();
    c->val_ = val;
    c->setDebugBlame(loc);
    return c;
}

class FloatConstNode : public ConstNode {
  public:
    double val_;
    void compHash() override;
    void inferDType() override;
    DEFINE_NODE_TRAIT(FloatConst);
};
typedef Ref<FloatConstNode> FloatConst;
inline Expr
makeFloatConst(double val,
               std::source_location loc = std::source_location::current()) {
    FloatConst c = FloatConst::make();
    c->val_ = val;
    c->setDebugBlame(loc);
    return c;
}

class BoolConstNode : public ConstNode {
  public:
    bool val_;
    void compHash() override;
    void inferDType() override;
    DEFINE_NODE_TRAIT(BoolConst);
};
typedef Ref<BoolConstNode> BoolConst;
inline Expr
makeBoolConst(bool val,
              std::source_location loc = std::source_location::current()) {
    BoolConst b = BoolConst::make();
    b->val_ = val;
    b->setDebugBlame(loc);
    return b;
}

class BinaryExprNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_ = ChildOf{this}, rhs_ = ChildOf{this};

    bool isBinary() const override { return true; }
    std::vector<Expr> children() const override { return {lhs_, rhs_}; }
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
    void inferDType() override;
    DEFINE_NODE_TRAIT(Add);
};
typedef Ref<AddNode> Add;
template <class T, class U>
Expr makeAdd(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Add a = Add::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class SubNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Sub);
};
typedef Ref<SubNode> Sub;
template <class T, class U>
Expr makeSub(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Sub a = Sub::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class MulNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Mul);
};
typedef Ref<MulNode> Mul;
template <class T, class U>
Expr makeMul(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Mul a = Mul::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/**
 * Floating-point division
 */
class RealDivNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(RealDiv);
};
typedef Ref<RealDivNode> RealDiv;
template <class T, class U>
Expr makeRealDiv(T &&lhs, U &&rhs,
                 std::source_location loc = std::source_location::current()) {
    RealDiv a = RealDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/**
 * Integer division, rounded towards negative infinity
 *
 * FloorDiv nodes are easy to analyze, and will be replaced by RoundTowards0Div
 * nodes before codegen if possible
 */
class FloorDivNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(FloorDiv);
};
typedef Ref<FloorDivNode> FloorDiv;
template <class T, class U>
Expr makeFloorDiv(T &&lhs, U &&rhs,
                  std::source_location loc = std::source_location::current()) {
    FloorDiv a = FloorDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/**
 * Integer division, rounded towards positive infinity
 *
 * CeilDiv nodes are easy to analyze, and will be replaced by RoundTowards0Div
 * nodes before codegen if possible
 */
class CeilDivNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(CeilDiv);
};
typedef Ref<CeilDivNode> CeilDiv;
template <class T, class U>
Expr makeCeilDiv(T &&lhs, U &&rhs,
                 std::source_location loc = std::source_location::current()) {
    CeilDiv a = CeilDiv::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/**
 * Integer division, rounded towards 0
 *
 * RoundTowards0Div nodes comply with the integer division behaviour in C. They
 * have minimal runtime overhead, but are hard to analyze
 */
class RoundTowards0DivNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(RoundTowards0Div);
};
typedef Ref<RoundTowards0DivNode> RoundTowards0Div;
template <class T, class U>
Expr makeRoundTowards0Div(
    T &&lhs, U &&rhs,
    std::source_location loc = std::source_location::current()) {
    RoundTowards0Div a = RoundTowards0Div::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/** Modulo
 *
 * Mod(3, 5) = 3
 * Mod(-3, 5) = 2
 * Mod(-3, -5) = -3
 * Mod(3, -5) = -2
 */
class ModNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Mod);
};
typedef Ref<ModNode> Mod;
template <class T, class U>
Expr makeMod(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Mod a = Mod::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

/** Remainder
 *
 * Remainder(3, 5) = 3
 * Remainder(-3, 5) = -3
 * Remainder(-3, -5) = -3
 * Remainder(3, -5) = 3
 */
class RemainderNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Remainder);
};
typedef Ref<RemainderNode> Remainder;
template <class T, class U>
Expr makeRemainder(T &&lhs, U &&rhs,
                   std::source_location loc = std::source_location::current()) {
    Remainder a = Remainder::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class MinNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Min);
};
typedef Ref<MinNode> Min;
template <class T, class U>
Expr makeMin(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Min m = Min::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    m->setDebugBlame(loc);
    return m;
}

class MaxNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Max);
};
typedef Ref<MaxNode> Max;
template <class T, class U>
Expr makeMax(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    Max m = Max::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    m->setDebugBlame(loc);
    return m;
}

class LTNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(LT);
};
typedef Ref<LTNode> LT;
template <class T, class U>
Expr makeLT(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    LT a = LT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class LENode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(LE);
};
typedef Ref<LENode> LE;
template <class T, class U>
Expr makeLE(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    LE a = LE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class GTNode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(GT);
};
typedef Ref<GTNode> GT;
template <class T, class U>
Expr makeGT(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    GT a = GT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class GENode : public NonCommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(GE);
};
typedef Ref<GENode> GE;
template <class T, class U>
Expr makeGE(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    GE a = GE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class EQNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(EQ);
};
typedef Ref<EQNode> EQ;
template <class T, class U>
Expr makeEQ(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    EQ a = EQ::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class NENode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
template <class T, class U>
Expr makeNE(T &&lhs, U &&rhs,
            std::source_location loc = std::source_location::current()) {
    NE a = NE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    a->setDebugBlame(loc);
    return a;
}

class LAndNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(LAnd);
};
typedef Ref<LAndNode> LAnd;
template <class T, class U>
Expr makeLAnd(T &&lhs, U &&rhs,
              std::source_location loc = std::source_location::current()) {
    LAnd l = LAnd::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    l->setDebugBlame(loc);
    return l;
}

class LOrNode : public CommutativeBinaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(LOr);
};
typedef Ref<LOrNode> LOr;
template <class T, class U>
Expr makeLOr(T &&lhs, U &&rhs,
             std::source_location loc = std::source_location::current()) {
    LOr l = LOr::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    l->setDebugBlame(loc);
    return l;
}

class UnaryExprNode : public ExprNode {
  public:
    SubTree<ExprNode> expr_ = ChildOf{this};

    void compHash() override;
    bool isUnary() const override { return true; }
    std::vector<Expr> children() const override { return {expr_}; }
};
typedef Ref<UnaryExprNode> UnaryExpr;

class LNotNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(LNot);
};
typedef Ref<LNotNode> LNot;
template <class T>
Expr makeLNot(T &&expr,
              std::source_location loc = std::source_location::current()) {
    LNot n = LNot::make();
    n->expr_ = std::forward<T>(expr);
    n->setDebugBlame(loc);
    return n;
}

class SqrtNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Sqrt);
};
typedef Ref<SqrtNode> Sqrt;
template <class T>
Expr makeSqrt(T &&expr,
              std::source_location loc = std::source_location::current()) {
    Sqrt s = Sqrt::make();
    s->expr_ = std::forward<T>(expr);
    s->setDebugBlame(loc);
    return s;
}

class ExpNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Exp);
};
typedef Ref<ExpNode> Exp;
template <class T>
Expr makeExp(T &&expr,
             std::source_location loc = std::source_location::current()) {
    Exp e = Exp::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class LnNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Ln);
};
typedef Ref<LnNode> Ln;
template <class T>
Expr makeLn(T &&expr,
            std::source_location loc = std::source_location::current()) {
    Ln e = Ln::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class SquareNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Square);
};
typedef Ref<SquareNode> Square;
template <class T>
Expr makeSquare(T &&expr,
                std::source_location loc = std::source_location::current()) {
    Square e = Square::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class SigmoidNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Sigmoid);
};
typedef Ref<SigmoidNode> Sigmoid;
template <class T>
Expr makeSigmoid(T &&expr,
                 std::source_location loc = std::source_location::current()) {
    Sigmoid e = Sigmoid::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class SinNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Sin);
};
typedef Ref<SinNode> Sin;
template <class T>
Expr makeSin(T &&expr,
             std::source_location loc = std::source_location::current()) {
    Sin e = Sin::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class CosNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Cos);
};
typedef Ref<CosNode> Cos;
template <class T>
Expr makeCos(T &&expr,
             std::source_location loc = std::source_location::current()) {
    Cos e = Cos::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class TanNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Tan);
};
typedef Ref<TanNode> Tan;
template <class T>
Expr makeTan(T &&expr,
             std::source_location loc = std::source_location::current()) {
    Tan e = Tan::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class TanhNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Tanh);
};
typedef Ref<TanhNode> Tanh;
template <class T>
Expr makeTanh(T &&expr,
              std::source_location loc = std::source_location::current()) {
    Tanh e = Tanh::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class AbsNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Abs);
};
typedef Ref<AbsNode> Abs;
template <class T>
Expr makeAbs(T &&expr,
             std::source_location loc = std::source_location::current()) {
    Abs e = Abs::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class FloorNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Floor);
};
typedef Ref<FloorNode> Floor;
template <class T>
Expr makeFloor(T &&expr,
               std::source_location loc = std::source_location::current()) {
    Floor e = Floor::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class CeilNode : public UnaryExprNode {
    void inferDType() override;
    DEFINE_NODE_TRAIT(Ceil);
};
typedef Ref<CeilNode> Ceil;
template <class T>
Expr makeCeil(T &&expr,
              std::source_location loc = std::source_location::current()) {
    Ceil e = Ceil::make();
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
    return e;
}

class IfExprNode : public ExprNode {
  public:
    SubTree<ExprNode> cond_ = ChildOf{this};
    SubTree<ExprNode> thenCase_ = ChildOf{this};
    SubTree<ExprNode> elseCase_ = ChildOf{this};
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override {
        return {cond_, thenCase_, elseCase_};
    }
    DEFINE_NODE_TRAIT(IfExpr);
};
typedef Ref<IfExprNode> IfExpr;
template <class T, class U, class V>
Expr makeIfExpr(T &&cond, U &&thenCase, V &&elseCase,
                std::source_location loc = std::source_location::current()) {
    IfExpr e = IfExpr::make();
    e->cond_ = std::forward<T>(cond);
    e->thenCase_ = std::forward<U>(thenCase);
    e->elseCase_ = std::forward<V>(elseCase);
    e->setDebugBlame(loc);
    return e;
}

class CastNode : public ExprNode {
  public:
    SubTree<ExprNode> expr_ = ChildOf{this};
    DataType destType_;
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override { return {expr_}; }
    DEFINE_NODE_TRAIT(Cast);
};
typedef Ref<CastNode> Cast;
template <class T>
Expr makeCast(T &&expr, DataType destType,
              std::source_location loc = std::source_location::current()) {
    Cast e = Cast::make();
    e->expr_ = std::forward<T>(expr);
    e->destType_ = destType;
    e->setDebugBlame(loc);
    return e;
}

/**
 * Invoke whatever target code
 */
class IntrinsicNode : public ExprNode {
  public:
    std::string
        format_; /// what to run. "%" is filled by parameters one by one
                 /// E.g. sinf(%). Use "%%" to escape for "%". If you need two
                 /// adjacent parameters, type "(%)(%)" or "% %".
    SubTreeList<ExprNode> params_ = ChildOf{this};
    DataType retType_;
    bool hasSideEffect_;
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override { return {params_}; }
    DEFINE_NODE_TRAIT(Intrinsic);
};
typedef Ref<IntrinsicNode> Intrinsic;
template <class T>
Expr makeIntrinsic(const std::string &format, T &&params, DataType retType,
                   bool hasSideEffect,
                   std::source_location loc = std::source_location::current()) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = std::forward<T>(params);
    i->retType_ = retType;
    i->hasSideEffect_ = hasSideEffect;
    i->setDebugBlame(loc);
    return i;
}
inline Expr
makeIntrinsic(const std::string &format, std::initializer_list<Expr> params,
              DataType retType, bool hasSideEffect,
              std::source_location loc = std::source_location::current()) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = params;
    i->retType_ = retType;
    i->hasSideEffect_ = hasSideEffect;
    i->setDebugBlame(loc);
    return i;
}

class LoadAtVersionNode : public ExprNode {
  public:
    std::string tapeName_;
    SubTreeList<ExprNode> indices_ = ChildOf{this};
    DataType loadType_;
    void compHash() override;
    void inferDType() override;
    std::vector<Expr> children() const override { return indices_; }
    DEFINE_NODE_TRAIT(LoadAtVersion);
};
typedef Ref<LoadAtVersionNode> LoadAtVersion;
template <class Tindices>
inline Expr
makeLoadAtVersion(const std::string &tapeName, Tindices &&indices,
                  const DataType loadType,
                  std::source_location loc = std::source_location::current()) {
    LoadAtVersion l = LoadAtVersion::make();
    l->tapeName_ = tapeName;
    l->indices_ = std::forward<Tindices>(indices);
    l->loadType_ = loadType;
    l->setDebugBlame(loc);
    return l;
}
inline Expr
makeLoadAtVersion(const std::string &tapeName, const std::vector<Expr> &indices,
                  const DataType loadType,
                  std::source_location loc = std::source_location::current()) {
    LoadAtVersion l = LoadAtVersion::make();
    l->tapeName_ = tapeName;
    l->indices_ = indices;
    l->loadType_ = loadType;
    l->setDebugBlame(loc);
    return l;
}

template <class T, class U>
Expr makeBinary(ASTNodeType nodeType, T &&lhs, U &&rhs,
                std::source_location loc = std::source_location::current()) {
    switch (nodeType) {
    case ASTNodeType::Add:
        return makeAdd(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::Sub:
        return makeSub(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::Mul:
        return makeMul(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::RealDiv:
        return makeRealDiv(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::FloorDiv:
        return makeFloorDiv(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::CeilDiv:
        return makeCeilDiv(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::RoundTowards0Div:
        return makeRoundTowards0Div(std::forward<T>(lhs), std::forward<U>(rhs),
                                    loc);
    case ASTNodeType::Mod:
        return makeMod(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::Remainder:
        return makeRemainder(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::Min:
        return makeMin(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::Max:
        return makeMax(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::LT:
        return makeLT(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::LE:
        return makeLE(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::GT:
        return makeGT(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::GE:
        return makeGE(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::EQ:
        return makeEQ(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::NE:
        return makeNE(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::LAnd:
        return makeLAnd(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    case ASTNodeType::LOr:
        return makeLOr(std::forward<T>(lhs), std::forward<U>(rhs), loc);
    default:
        ASSERT(false);
    }
}

template <class T>
Expr makeUnary(ASTNodeType nodeType, T &&expr,
               std::source_location loc = std::source_location::current()) {
    switch (nodeType) {
    case ASTNodeType::LNot:
        return makeLNot(std::forward<T>(expr), loc);
    case ASTNodeType::Sqrt:
        return makeSqrt(std::forward<T>(expr), loc);
    case ASTNodeType::Exp:
        return makeExp(std::forward<T>(expr), loc);
    case ASTNodeType::Ln:
        return makeLn(std::forward<T>(expr), loc);
    case ASTNodeType::Square:
        return makeSquare(std::forward<T>(expr), loc);
    case ASTNodeType::Sigmoid:
        return makeSigmoid(std::forward<T>(expr), loc);
    case ASTNodeType::Sin:
        return makeSin(std::forward<T>(expr), loc);
    case ASTNodeType::Cos:
        return makeCos(std::forward<T>(expr), loc);
    case ASTNodeType::Tan:
        return makeTan(std::forward<T>(expr), loc);
    case ASTNodeType::Tanh:
        return makeTanh(std::forward<T>(expr), loc);
    case ASTNodeType::Abs:
        return makeAbs(std::forward<T>(expr), loc);
    case ASTNodeType::Floor:
        return makeFloor(std::forward<T>(expr), loc);
    case ASTNodeType::Ceil:
        return makeCeil(std::forward<T>(expr), loc);
    default:
        ASSERT(false);
    }
}

} // namespace freetensor

#endif // FREE_TENSOR_EXPR_H
