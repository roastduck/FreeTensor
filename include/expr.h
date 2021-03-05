#ifndef EXPR_H
#define EXPR_H

#include <string>
#include <vector>

#include <ast.h>

namespace ir {

class ExprNode : public ASTNode {
    DEFINE_NODE_ACCESS(Expr);
};
typedef Ref<ExprNode> Expr;

/**
 * Matches any expression
 *
 * Only used in pattern matching
 */
class AnyExprNode : public ExprNode {
    DEFINE_NODE_TRAIT(AnyExpr);
};
typedef Ref<AnyExprNode> AnyExpr;
inline Expr makeAnyExpr() { return AnyExpr::make(); }

class VarNode : public ExprNode {
  public:
    std::string name_;
    DEFINE_NODE_TRAIT(Var);
};
typedef Ref<VarNode> Var;
inline Expr makeVar(const std::string &name) {
    Var v = Var::make();
    v->name_ = name;
    return v;
}

class LoadNode : public ExprNode {
  public:
    std::string var_;
    std::vector<Expr> indices_;
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
inline Expr makeLoad(const std::string &var, const std::vector<Expr> &indices) {
    Load l = Load::make();
    l->var_ = var, l->indices_ = indices;
    return l;
}

class IntConstNode : public ExprNode {
  public:
    int val_;
    DEFINE_NODE_TRAIT(IntConst);
};
typedef Ref<IntConstNode> IntConst;
inline Expr makeIntConst(int val) {
    IntConst c = IntConst::make();
    c->val_ = val;
    return c;
}

class FloatConstNode : public ExprNode {
  public:
    double val_;
    DEFINE_NODE_TRAIT(FloatConst);
};
typedef Ref<FloatConstNode> FloatConst;
inline Expr makeFloatConst(double val) {
    FloatConst c = FloatConst::make();
    c->val_ = val;
    return c;
}

class BoolConstNode : public ExprNode {
  public:
    bool val_;
    DEFINE_NODE_TRAIT(BoolConst);
};
typedef Ref<BoolConstNode> BoolConst;
inline Expr makeBoolConst(bool val) {
    BoolConst b = BoolConst::make();
    b->val_ = val;
    return b;
}

class AddNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Add);
};
typedef Ref<AddNode> Add;
template <class T, class U> Expr makeAdd(T &&lhs, U &&rhs) {
    Add a = Add::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class SubNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Sub);
};
typedef Ref<SubNode> Sub;
template <class T, class U> Expr makeSub(T &&lhs, U &&rhs) {
    Sub a = Sub::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MulNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Mul);
};
typedef Ref<MulNode> Mul;
template <class T, class U> Expr makeMul(T &&lhs, U &&rhs) {
    Mul a = Mul::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

/**
 * Floating-point division
 */
class RealDivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(RealDiv);
};
typedef Ref<RealDivNode> RealDiv;
template <class T, class U> Expr makeRealDiv(T &&lhs, U &&rhs) {
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
class FloorDivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(FloorDiv);
};
typedef Ref<FloorDivNode> FloorDiv;
template <class T, class U> Expr makeFloorDiv(T &&lhs, U &&rhs) {
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
class CeilDivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(CeilDiv);
};
typedef Ref<CeilDivNode> CeilDiv;
template <class T, class U> Expr makeCeilDiv(T &&lhs, U &&rhs) {
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
class RoundTowards0DivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(RoundTowards0Div);
};
typedef Ref<RoundTowards0DivNode> RoundTowards0Div;
template <class T, class U> Expr makeRoundTowards0Div(T &&lhs, U &&rhs) {
    RoundTowards0Div a = RoundTowards0Div::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class ModNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Mod);
};
typedef Ref<ModNode> Mod;
template <class T, class U> Expr makeMod(T &&lhs, U &&rhs) {
    Mod a = Mod::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MinNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Min);
};
typedef Ref<MinNode> Min;
template <class T, class U> Expr makeMin(T &&lhs, U &&rhs) {
    Min m = Min::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class MaxNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Max);
};
typedef Ref<MaxNode> Max;
template <class T, class U> Expr makeMax(T &&lhs, U &&rhs) {
    Max m = Max::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class LTNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LT);
};
typedef Ref<LTNode> LT;
template <class T, class U> Expr makeLT(T &&lhs, U &&rhs) {
    LT a = LT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LE);
};
typedef Ref<LENode> LE;
template <class T, class U> Expr makeLE(T &&lhs, U &&rhs) {
    LE a = LE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GTNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(GT);
};
typedef Ref<GTNode> GT;
template <class T, class U> Expr makeGT(T &&lhs, U &&rhs) {
    GT a = GT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(GE);
};
typedef Ref<GENode> GE;
template <class T, class U> Expr makeGE(T &&lhs, U &&rhs) {
    GE a = GE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class EQNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(EQ);
};
typedef Ref<EQNode> EQ;
template <class T, class U> Expr makeEQ(T &&lhs, U &&rhs) {
    EQ a = EQ::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class NENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
template <class T, class U> Expr makeNE(T &&lhs, U &&rhs) {
    NE a = NE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LAndNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LAnd);
};
typedef Ref<LAndNode> LAnd;
template <class T, class U> Expr makeLAnd(T &&lhs, U &&rhs) {
    LAnd l = LAnd::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class LOrNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LOr);
};
typedef Ref<LOrNode> LOr;
template <class T, class U> Expr makeLOr(T &&lhs, U &&rhs) {
    LOr l = LOr::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class LNotNode : public ExprNode {
  public:
    Expr expr_;
    DEFINE_NODE_TRAIT(LNot);
};
typedef Ref<LNotNode> LNot;
template <class T> Expr makeLNot(T &&expr) {
    LNot n = LNot::make();
    n->expr_ = std::forward<T>(expr);
    return n;
}

/**
 * Invoke whatever target code
 */
class IntrinsicNode : public ExprNode {
  public:
    std::string format_; /// what to run. "%" is filled by parameters one by one
                         /// E.g. sinf(%)
    std::vector<Expr> params_;
    DEFINE_NODE_TRAIT(Intrinsic);
};
typedef Ref<IntrinsicNode> Intrinsic;
template <class T> Expr makeIntrinsic(const std::string &format, T &&params) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = std::forward<T>(params);
    return i;
}
inline Expr makeIntrinsic(const std::string &format,
                          std::initializer_list<Expr> params) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = params;
    return i;
}

} // namespace ir

#endif // EXPR_H
