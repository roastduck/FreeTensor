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
    Expr var_;
    std::vector<Expr> indices_;
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
inline Expr makeLoad(const Expr &var, const std::vector<Expr> &indices) {
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
inline Expr makeAdd(const Expr &lhs, const Expr &rhs) {
    Add a = Add::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class SubNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Sub);
};
typedef Ref<SubNode> Sub;
inline Expr makeSub(const Expr &lhs, const Expr &rhs) {
    Sub a = Sub::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class MulNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Mul);
};
typedef Ref<MulNode> Mul;
inline Expr makeMul(const Expr &lhs, const Expr &rhs) {
    Mul a = Mul::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class DivNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Div);
};
typedef Ref<DivNode> Div;
inline Expr makeDiv(const Expr &lhs, const Expr &rhs) {
    Div a = Div::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class ModNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(Mod);
};
typedef Ref<ModNode> Mod;
inline Expr makeMod(const Expr &lhs, const Expr &rhs) {
    Mod a = Mod::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class LTNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LT);
};
typedef Ref<LTNode> LT;
inline Expr makeLT(const Expr &lhs, const Expr &rhs) {
    LT a = LT::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class LENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(LE);
};
typedef Ref<LENode> LE;
inline Expr makeLE(const Expr &lhs, const Expr &rhs) {
    LE a = LE::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class GTNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(GT);
};
typedef Ref<GTNode> GT;
inline Expr makeGT(const Expr &lhs, const Expr &rhs) {
    GT a = GT::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class GENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(GE);
};
typedef Ref<GENode> GE;
inline Expr makeGE(const Expr &lhs, const Expr &rhs) {
    GE a = GE::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class EQNode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(EQ);
};
typedef Ref<EQNode> EQ;
inline Expr makeEQ(const Expr &lhs, const Expr &rhs) {
    EQ a = EQ::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

class NENode : public ExprNode {
  public:
    Expr lhs_, rhs_;
    DEFINE_NODE_TRAIT(NE);
};
typedef Ref<NENode> NE;
inline Expr makeNE(const Expr &lhs, const Expr &rhs) {
    NE a = NE::make();
    a->lhs_ = lhs, a->rhs_ = rhs;
    return a;
}

} // namespace ir

#endif // EXPR_H
