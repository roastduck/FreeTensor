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
    DEFINE_NODE_TRAIT(AnyExpr);
};
typedef Ref<AnyExprNode> AnyExpr;
#define makeAnyExpr(...) makeNode(AnyExpr, __VA_ARGS__)
inline Expr _makeAnyExpr() { return AnyExpr::make(); }

class VarNode : public ExprNode {
  public:
    std::string name_;
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
    std::vector<SubTree<ExprNode>> indices_;
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
#define makeLoad(...) makeNode(Load, __VA_ARGS__)
inline Expr _makeLoad(const std::string &var,
                      const std::vector<Expr> &indices) {
    Load l = Load::make();
    l->var_ = var;
    l->indices_ =
        std::vector<SubTree<ExprNode>>(indices.begin(), indices.end());
    return l;
}

class IntConstNode : public ExprNode {
  public:
    int val_;
    DEFINE_NODE_TRAIT(IntConst);
};
typedef Ref<IntConstNode> IntConst;
#define makeIntConst(...) makeNode(IntConst, __VA_ARGS__)
inline Expr _makeIntConst(int val) {
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
#define makeFloatConst(...) makeNode(FloatConst, __VA_ARGS__)
inline Expr _makeFloatConst(double val) {
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
#define makeBoolConst(...) makeNode(BoolConst, __VA_ARGS__)
inline Expr _makeBoolConst(bool val) {
    BoolConst b = BoolConst::make();
    b->val_ = val;
    return b;
}

class AddNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(Add);
};
typedef Ref<AddNode> Add;
#define makeAdd(...) makeNode(Add, __VA_ARGS__)
template <class T, class U> Expr _makeAdd(T &&lhs, U &&rhs) {
    Add a = Add::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class SubNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(Sub);
};
typedef Ref<SubNode> Sub;
#define makeSub(...) makeNode(Sub, __VA_ARGS__)
template <class T, class U> Expr _makeSub(T &&lhs, U &&rhs) {
    Sub a = Sub::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MulNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
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
class RealDivNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
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
class FloorDivNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
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
class CeilDivNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
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
class RoundTowards0DivNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(RoundTowards0Div);
};
typedef Ref<RoundTowards0DivNode> RoundTowards0Div;
#define makeRoundTowards0Div(...) makeNode(RoundTowards0Div, __VA_ARGS__)
template <class T, class U> Expr _makeRoundTowards0Div(T &&lhs, U &&rhs) {
    RoundTowards0Div a = RoundTowards0Div::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

// FIXME: Deal with negative numbers in Mod
class ModNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(Mod);
};
typedef Ref<ModNode> Mod;
#define makeMod(...) makeNode(Mod, __VA_ARGS__)
template <class T, class U> Expr _makeMod(T &&lhs, U &&rhs) {
    Mod a = Mod::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class MinNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(Min);
};
typedef Ref<MinNode> Min;
#define makeMin(...) makeNode(Min, __VA_ARGS__)
template <class T, class U> Expr _makeMin(T &&lhs, U &&rhs) {
    Min m = Min::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class MaxNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(Max);
};
typedef Ref<MaxNode> Max;
#define makeMax(...) makeNode(Max, __VA_ARGS__)
template <class T, class U> Expr _makeMax(T &&lhs, U &&rhs) {
    Max m = Max::make();
    m->lhs_ = std::forward<T>(lhs), m->rhs_ = std::forward<U>(rhs);
    return m;
}

class LTNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(LT);
};
typedef Ref<LTNode> LT;
#define makeLT(...) makeNode(LT, __VA_ARGS__)
template <class T, class U> Expr _makeLT(T &&lhs, U &&rhs) {
    LT a = LT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LENode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(LE);
};
typedef Ref<LENode> LE;
#define makeLE(...) makeNode(LE, __VA_ARGS__)
template <class T, class U> Expr _makeLE(T &&lhs, U &&rhs) {
    LE a = LE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GTNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(GT);
};
typedef Ref<GTNode> GT;
#define makeGT(...) makeNode(GT, __VA_ARGS__)
template <class T, class U> Expr _makeGT(T &&lhs, U &&rhs) {
    GT a = GT::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class GENode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(GE);
};
typedef Ref<GENode> GE;
#define makeGE(...) makeNode(GE, __VA_ARGS__)
template <class T, class U> Expr _makeGE(T &&lhs, U &&rhs) {
    GE a = GE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class EQNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(EQ);
};
typedef Ref<EQNode> EQ;
#define makeEQ(...) makeNode(EQ, __VA_ARGS__)
template <class T, class U> Expr _makeEQ(T &&lhs, U &&rhs) {
    EQ a = EQ::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class NENode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
#define makeNE(...) makeNode(NE, __VA_ARGS__)
template <class T, class U> Expr _makeNE(T &&lhs, U &&rhs) {
    NE a = NE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class LAndNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(LAnd);
};
typedef Ref<LAndNode> LAnd;
#define makeLAnd(...) makeNode(LAnd, __VA_ARGS__)
template <class T, class U> Expr _makeLAnd(T &&lhs, U &&rhs) {
    LAnd l = LAnd::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class LOrNode : public ExprNode {
  public:
    SubTree<ExprNode> lhs_, rhs_;
    DEFINE_NODE_TRAIT(LOr);
};
typedef Ref<LOrNode> LOr;
#define makeLOr(...) makeNode(LOr, __VA_ARGS__)
template <class T, class U> Expr _makeLOr(T &&lhs, U &&rhs) {
    LOr l = LOr::make();
    l->lhs_ = std::forward<T>(lhs), l->rhs_ = std::forward<U>(rhs);
    return l;
}

class LNotNode : public ExprNode {
  public:
    SubTree<ExprNode> expr_;
    DEFINE_NODE_TRAIT(LNot);
};
typedef Ref<LNotNode> LNot;
#define makeLNot(...) makeNode(LNot, __VA_ARGS__)
template <class T> Expr _makeLNot(T &&expr) {
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
    std::vector<SubTree<ExprNode>> params_;
    DataType retType_;
    DEFINE_NODE_TRAIT(Intrinsic);
};
typedef Ref<IntrinsicNode> Intrinsic;
#define makeIntrinsic(...) makeNode(Intrinsic, __VA_ARGS__)
template <class T>
Expr _makeIntrinsic(const std::string &format, T &&params, DataType retType) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = std::vector<SubTree<ExprNode>>(params.begin(), params.end());
    i->retType_ = retType;
    return i;
}
inline Expr _makeIntrinsic(const std::string &format,
                           std::initializer_list<Expr> params,
                           DataType retType) {
    Intrinsic i = Intrinsic::make();
    i->format_ = format;
    i->params_ = std::vector<SubTree<ExprNode>>(params.begin(), params.end());
    i->retType_ = retType;
    return i;
}

} // namespace ir

#endif // EXPR_H
