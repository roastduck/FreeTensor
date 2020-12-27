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

class VarNode : public ExprNode {
  public:
    const std::string name_; // Multiple expressions may share the same VarNode
                             // reference, so const
    DEFINE_NODE_TRAIT(Var);
};
typedef Ref<VarNode> Var;
inline Expr makeVar(const std::string &name) {
    Var v = Var::make();
    const_cast<std::string &>(v->name_) = name;
    return v;
}

class LoadNode : public ExprNode {
  public:
    std::string var_;
    std::vector<Expr> indices_;

    std::vector<std::vector<Expr>> info_dep_rw_; // RAW and WAR dependencies

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

class DivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Div);
};
typedef Ref<DivNode> Div;
template <class T, class U> Expr makeDiv(T &&lhs, U &&rhs) {
    Div a = Div::make();
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

class LTNode : public ExprNode {
  public:
    Expr lhs_, rhs_;

    Expr info_norm_form_; // this <==> (info_norm_form_ < 0)

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

    Expr info_norm_form_; // this <==> (info_norm_form_ <= 0)

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

    Expr info_norm_form_; // this <==> (info_norm_form_ > 0)

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

    Expr info_norm_form_; // this <==> (info_norm_form_ >= 0)

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

    Expr info_norm_form_; // this <==> (info_norm_form_ == 0)

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

    Expr info_norm_form_; // this <==> (info_norm_form_ != 0)

    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
template <class T, class U> Expr makeNE(T &&lhs, U &&rhs) {
    NE a = NE::make();
    a->lhs_ = std::forward<T>(lhs), a->rhs_ = std::forward<U>(rhs);
    return a;
}

class NotNode : public ExprNode {
  public:
    Expr expr_;
    DEFINE_NODE_TRAIT(Not);
};
typedef Ref<NotNode> Not;
template <class T> Expr makeNot(T &&expr) {
    Not n = Not::make();
    n->expr_ = std::forward<T>(expr);
    return n;
}

} // namespace ir

#endif // EXPR_H
