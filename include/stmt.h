#ifndef STMT_H
#define STMT_H

#include <string>
#include <vector>

#include <ast.h>
#include <buffer.h>

namespace ir {

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
    std::vector<SubTree<StmtNode>> stmts_;
    DEFINE_NODE_TRAIT(StmtSeq);
};
typedef Ref<StmtSeqNode> StmtSeq;
template <class Tstmts>
Stmt makeStmtSeq(const std::string &id, Tstmts &&stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = std::vector<SubTree<StmtNode>>(stmts.begin(), stmts.end());
    return s;
}
inline Stmt makeStmtSeq(const std::string &id,
                        std::initializer_list<Stmt> stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = std::vector<SubTree<StmtNode>>(stmts.begin(), stmts.end());
    return s;
}

class VarDefNode : public StmtNode {
  public:
    std::string name_;
    Ref<Buffer> buffer_;
    SubTree<ExprNode, NullPolicy::Nullable>
        sizeLim_; // limit the buffer size to a specific
                  // expression, other than the size of buffer_
    SubTree<StmtNode> body_;
    bool pinned_; // If pinned, SinkVar and ShrinkVar will not alter this node

    VarDefNode(const VarDefNode &other);            // Deep copy buffer_
    VarDefNode &operator=(const VarDefNode &other); // Deep copy buffer_

    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
template <class Tbuffer, class Tbody>
Stmt makeVarDef(const std::string &id, const std::string &name,
                Tbuffer &&buffer, const Expr &sizeLim, Tbody &&body,
                bool pinned) {
    VarDef d = VarDef::make();
    d->setId(id);
    d->name_ = name;
    d->buffer_ = Ref<Buffer>::make(std::forward<Tbuffer>(buffer));
    d->sizeLim_ = sizeLim;
    d->body_ = std::forward<Tbody>(body);
    d->pinned_ = pinned;
    return d;
}

class StoreNode : public StmtNode {
  public:
    std::string var_;
    std::vector<SubTree<ExprNode>> indices_;
    SubTree<ExprNode> expr_;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
template <class Tindices, class Texpr>
Stmt makeStore(const std::string &id, const std::string &var,
               Tindices &&indices, Texpr &&expr) {
    Store s = Store::make();
    s->setId(id);
    s->var_ = var;
    s->indices_ =
        std::vector<SubTree<ExprNode>>(indices.begin(), indices.end());
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}

enum class ReduceOp : int { Add, Min, Max };
class ReduceToNode : public StmtNode {
  public:
    std::string var_;
    std::vector<SubTree<ExprNode>> indices_;
    ReduceOp op_;
    SubTree<ExprNode> expr_;
    bool atomic_;
    DEFINE_NODE_TRAIT(ReduceTo)
};
typedef Ref<ReduceToNode> ReduceTo;
template <class Tindices, class Texpr>
Stmt makeReduceTo(const std::string &id, const std::string &var,
                  Tindices &&indices, ReduceOp op, Texpr &&expr, bool atomic) {
    ReduceTo a = ReduceTo::make();
    a->setId(id);
    a->var_ = var;
    a->indices_ =
        std::vector<SubTree<ExprNode>>(indices.begin(), indices.end());
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->atomic_ = atomic;
    return a;
}

class ForNode : public StmtNode {
  public:
    std::string iter_;
    SubTree<ExprNode> begin_, end_;
    std::string parallel_;
    bool unroll_;
    SubTree<StmtNode> body_;

    SubTree<ExprNode, NullPolicy::Nullable> infoLen_;

    DEFINE_NODE_TRAIT(For);
};
typedef Ref<ForNode> For;
template <class Tbegin, class Tend, class Tbody>
Stmt makeFor(const std::string &id, const std::string &iter, Tbegin &&begin,
             Tend &&end, const std::string &parallel, const bool unroll,
             Tbody &&body) {
    For f = For::make();
    f->setId(id);
    f->iter_ = iter;
    f->begin_ = std::forward<Tbegin>(begin);
    f->end_ = std::forward<Tend>(end);
    f->parallel_ = parallel;
    f->body_ = std::forward<Tbody>(body);
    f->unroll_ = unroll;
    return f;
}

class IfNode : public StmtNode {
  public:
    SubTree<ExprNode> cond_;
    SubTree<StmtNode> thenCase_;
    SubTree<StmtNode, NullPolicy::Nullable> elseCase_;

    SubTree<ExprNode, NullPolicy::Nullable> infoNotCond_;

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
    SubTree<ExprNode> cond_;
    SubTree<StmtNode> body_;
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

/**
 * Evaluate an expression and do nothing else
 *
 * Can be used to call an intrinsic
 */
class EvalNode : public StmtNode {
  public:
    SubTree<ExprNode> expr_;
    DEFINE_NODE_TRAIT(Eval);
};
typedef Ref<EvalNode> Eval;
template <class T> Stmt makeEval(const std::string &id, T &&expr) {
    Eval e = Eval::make();
    e->setId(id);
    e->expr_ = std::forward<T>(expr);
    return e;
}

} // namespace ir

#endif // STMT_H
