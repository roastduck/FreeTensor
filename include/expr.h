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

} // namespace ir

#endif // EXPR_H
