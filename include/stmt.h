#ifndef STMT_H
#define STMT_H

#include <string>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <expr.h>

namespace ir {

class StmtNode : public ASTNode {
    std::string id_;
    static uint64_t idCnt_;

  public:
    void setId(const std::string &id);
    const std::string &id() const;
    bool hasNamedId() const;

    DEFINE_NODE_ACCESS(Stmt);
};
typedef Ref<StmtNode> Stmt;

/**
 * Matches any statement
 *
 * Only used in pattern matching
 */
class AnyNode : public StmtNode {
    DEFINE_NODE_TRAIT(Any);
};
typedef Ref<AnyNode> Any;
inline Stmt makeAny() { return Any::make(); }

class StmtSeqNode : public StmtNode {
  public:
    std::vector<Stmt> stmts_;
    DEFINE_NODE_TRAIT(StmtSeq);
};
typedef Ref<StmtSeqNode> StmtSeq;
template <class Tstmts>
Stmt makeStmtSeq(const std::string &id, Tstmts &&stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = std::forward<Tstmts>(stmts);
    return s;
}
inline Stmt makeStmtSeq(const std::string &id,
                        std::initializer_list<Stmt> stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = stmts;
    return s;
}

class VarDefNode : public StmtNode {
  public:
    std::string name_;
    Ref<Buffer> buffer_;
    Stmt body_;

    Ref<std::vector<Expr>> info_acc_lower_; // lower_bound(access)
    Ref<std::vector<Expr>> info_acc_len_;   // upper_bound(access_i - access_j)

    VarDefNode(const VarDefNode &other);            // Deep copy
    VarDefNode &operator=(const VarDefNode &other); // Deep copy

    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
template <class Tbuffer, class Tbody>
Stmt makeVarDef(const std::string &id, const std::string &name,
                Tbuffer &&buffer, Tbody &&body) {
    VarDef d = VarDef::make();
    d->setId(id);
    d->name_ = name;
    d->buffer_ = Ref<Buffer>::make(std::forward<Tbuffer>(buffer));
    d->body_ = std::forward<Tbody>(body);
    return d;
}

class StoreNode : public StmtNode {
  public:
    std::string var_;
    std::vector<Expr> indices_;
    Expr expr_;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
template <class Tindices, class Texpr>
Stmt makeStore(const std::string &id, const std::string &var,
               Tindices &&indices, Texpr &&expr) {
    Store s = Store::make();
    s->setId(id);
    s->var_ = var;
    s->indices_ = std::forward<Tindices>(indices);
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}

class AddToNode : public StmtNode {
  public:
    std::string var_;
    std::vector<Expr> indices_;
    Expr expr_;
    DEFINE_NODE_TRAIT(AddTo)
};
typedef Ref<AddToNode> AddTo;
template <class Tindices, class Texpr>
Stmt makeAddTo(const std::string &id, const std::string &var,
               Tindices &&indices, Texpr &&expr) {
    AddTo a = AddTo::make();
    a->setId(id);
    a->var_ = var;
    a->indices_ = std::forward<Tindices>(indices);
    a->expr_ = std::forward<Texpr>(expr);
    return a;
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
Stmt makeFor(const std::string &id, const std::string &iter, Tbegin &&begin,
             Tend &&end, Tbody &&body) {
    For f = For::make();
    f->setId(id);
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
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt makeIf(const std::string &id, Tcond &&cond, Tthen &&thenCase,
            Telse &&elseCase = nullptr) {
    If i = If::make();
    i->setId(id);
    i->cond_ = std::forward<Tcond>(cond);
    i->thenCase_ = std::forward<Tthen>(thenCase);
    i->elseCase_ = std::forward<Telse>(elseCase);
    return i;
}

class AssertNode : public StmtNode {
  public:
    Expr cond_;
    Stmt body_;
    DEFINE_NODE_TRAIT(Assert);
};
typedef Ref<AssertNode> Assert;
template <class Tcond, class Tbody>
Stmt makeAssert(const std::string &id, Tcond &&cond, Tbody &&body) {
    Assert a = Assert::make();
    a->setId(id);
    a->cond_ = std::forward<Tcond>(cond);
    a->body_ = std::forward<Tbody>(body);
    return a;
}

} // namespace ir

#endif // STMT_H
