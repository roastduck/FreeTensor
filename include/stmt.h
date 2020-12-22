#ifndef STMT_H
#define STMT_H

#include <string>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <expr.h>

namespace ir {

class StmtNode : public ASTNode {
    DEFINE_NODE_ACCESS(Stmt);
};
typedef Ref<StmtNode> Stmt;

class StmtSeqNode : public StmtNode {
  public:
    std::vector<Stmt> stmts_;
    DEFINE_NODE_TRAIT(StmtSeq);
};
typedef Ref<StmtSeqNode> StmtSeq;
template <class Tstmts> Stmt makeStmtSeq(Tstmts &&stmts) {
    StmtSeq s = StmtSeq::make();
    s->stmts_ = std::forward<Tstmts>(stmts);
    return s;
}

class VarDefNode : public StmtNode {
  public:
    std::string name_;
    Ref<Buffer> buffer_;
    Stmt body_;

    VarDefNode(const VarDefNode &other);            // Deep copy
    VarDefNode &operator=(const VarDefNode &other); // Deep copy

    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
template <class Tbuffer, class Tbody>
Stmt makeVarDef(const std::string &name, Tbuffer &&buffer, Tbody &&body) {
    VarDef d = VarDef::make();
    d->name_ = name;
    d->buffer_ = Ref<Buffer>::make(std::forward<Tbuffer>(buffer));
    d->body_ = std::forward<Tbody>(body);
    return d;
}

class StoreNode : public StmtNode {
  public:
    Expr var_;
    std::vector<Expr> indices_;
    Expr expr_;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
template <class Tvar, class Tindices, class Texpr>
Stmt makeStore(Tvar &&var, Tindices &&indices, Texpr &&expr) {
    Store s = Store::make();
    s->var_ = std::forward<Tvar>(var);
    s->indices_ = std::forward<Tindices>(indices),
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}

class ForNode : public StmtNode {
  public:
    std::string iter_;
    Expr begin_, end_;
    Stmt body_;
    DEFINE_NODE_TRAIT(For);
};
typedef Ref<ForNode> For;
template <class Tbegin, class Tend, class Tbody>
Stmt makeFor(const std::string &iter, Tbegin &&begin, Tend &&end,
             Tbody &&body) {
    For f = For::make();
    f->iter_ = iter;
    f->begin_ = std::forward<Tbegin>(begin);
    f->end_ = std::forward<Tend>(end);
    f->body_ = std::forward<Tbody>(body);
    return f;
}

class IfNode : public StmtNode {
  public:
    Expr cond_;
    Stmt thenCase_, elseCase_;
    DEFINE_NODE_TRAIT(If);
};
typedef Ref<IfNode> If;
template <class Tcond, class Tthen, class Telse>
Stmt makeIf(Tcond &&cond, Tthen &&thenCase, Telse &&elseCase = nullptr) {
    If i = If::make();
    i->cond_ = std::forward<Tcond>(cond);
    i->thenCase_ = std::forward<Tthen>(thenCase);
    i->elseCase_ = std::forward<Telse>(elseCase);
    return i;
}

} // namespace ir

#endif // STMT_H
