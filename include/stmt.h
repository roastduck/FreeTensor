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
inline Stmt makeStmtSeq(const std::vector<Stmt> &stmts) {
    StmtSeq s = StmtSeq::make();
    s->stmts_ = stmts;
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
inline Stmt makeVarDef(const std::string &name, const Buffer &buffer,
                       const Stmt &body) {
    VarDef d = VarDef::make();
    d->name_ = name, d->buffer_ = Ref<Buffer>::make(buffer), d->body_ = body;
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
inline Stmt makeStore(const Expr &var, const std::vector<Expr> &indices,
                      const Expr &expr) {
    Store s = Store::make();
    s->var_ = var, s->indices_ = indices, s->expr_ = expr;
    return s;
}

} // namespace ir

#endif // STMT_H
