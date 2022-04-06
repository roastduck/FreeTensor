#ifndef STMT_H
#define STMT_H

#include <string>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <parallel_scope.h>

namespace ir {

/**
 * Matches any statement
 *
 * Only used in pattern matching
 */
class AnyNode : public StmtNode {
  public:
    void compHash() override;
    DEFINE_NODE_TRAIT(Any);
};
typedef Ref<AnyNode> Any;
#define makeAny(...) makeNode(Any, __VA_ARGS__)
inline Stmt _makeAny() { return Any::make(); }

class StmtSeqNode : public StmtNode {
  public:
    SubTreeList<StmtNode> stmts_;
    void compHash() override;
    DEFINE_NODE_TRAIT(StmtSeq);
};
typedef Ref<StmtSeqNode> StmtSeq;
#define makeStmtSeq(...) makeNode(StmtSeq, __VA_ARGS__)
template <class Tstmts> Stmt _makeStmtSeq(const ID &id, Tstmts &&stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = std::forward<Tstmts>(stmts);
    return s;
}
inline Stmt _makeStmtSeq(const ID &id, std::initializer_list<Stmt> stmts) {
    StmtSeq s = StmtSeq::make();
    s->setId(id);
    s->stmts_ = stmts;
    return s;
}

class VarDefNode : public StmtNode {
  public:
    std::string name_;
    SubTree<Buffer> buffer_;
    SubTree<ExprNode, NullPolicy::Nullable>
        sizeLim_; // limit the buffer size to a specific
                  // expression, other than the size of buffer_
    SubTree<StmtNode> body_;
    bool pinned_; // If pinned, SinkVar and ShrinkVar will not alter this node
    void compHash() override;
    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
#define makeVarDef(...) makeNode(VarDef, __VA_ARGS__)
template <class Tbuffer, class Tbody>
Stmt _makeVarDef(const ID &id, const std::string &name, Tbuffer &&buffer,
                 const Expr &sizeLim, Tbody &&body, bool pinned) {
    VarDef d = VarDef::make();
    d->setId(id);
    d->name_ = name;
    auto tmp = SubTree<Buffer>(buffer);
    d->buffer_ = std::move(tmp);
    // d->buffer_ = std::forward<Tbuffer>(buffer);
    d->sizeLim_ = sizeLim;
    d->body_ = std::forward<Tbody>(body);
    d->pinned_ = pinned;
    return d;
}

class StoreNode : public StmtNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_;
    SubTree<ExprNode> expr_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
#define makeStore(...) makeNode(Store, __VA_ARGS__)
template <class Tindices, class Texpr>
Stmt _makeStore(const ID &id, const std::string &var, Tindices &&indices,
                Texpr &&expr) {
    Store s = Store::make();
    s->setId(id);
    s->var_ = var;
    s->indices_ = std::forward<Tindices>(indices);
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}
template <class Texpr>
Stmt _makeStore(const ID &id, const std::string &var,
                const std::vector<Expr> &indices, Texpr &&expr) {
    Store s = Store::make();
    s->setId(id);
    s->var_ = var;
    s->indices_ = indices;
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}

enum class ReduceOp : int { Add, Mul, Min, Max };
class ReduceToNode : public StmtNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_;
    ReduceOp op_;
    SubTree<ExprNode> expr_;
    bool atomic_;
    void compHash() override;
    DEFINE_NODE_TRAIT(ReduceTo)
};
typedef Ref<ReduceToNode> ReduceTo;
#define makeReduceTo(...) makeNode(ReduceTo, __VA_ARGS__)
template <class Tindices, class Texpr>
Stmt _makeReduceTo(const ID &id, const std::string &var, Tindices &&indices,
                   ReduceOp op, Texpr &&expr, bool atomic) {
    ReduceTo a = ReduceTo::make();
    a->setId(id);
    a->var_ = var;
    a->indices_ = std::forward<Tindices>(indices);
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->atomic_ = atomic;
    return a;
}
template <class Texpr>
Stmt _makeReduceTo(const ID &id, const std::string &var,
                   const std::vector<Expr> &indices, ReduceOp op, Texpr &&expr,
                   bool atomic) {
    ReduceTo a = ReduceTo::make();
    a->setId(id);
    a->var_ = var;
    a->indices_ = indices;
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->atomic_ = atomic;
    return a;
}

struct ReductionItem {
    ReduceOp op_;
    std::string var_;
    SubTreeList<ExprNode> begins_, ends_;
};

struct ForProperty {
    ParallelScope parallel_;
    bool unroll_, vectorize_;
    std::vector<ReductionItem> reductions_;
    std::vector<std::string> noDeps_; // vars that are explicitly marked to have
                                      // no dependencies over this loop
    bool preferLibs_; // Aggresively transform to external library calls in
                      // auto-schedule

    ForProperty()
        : parallel_(), unroll_(false), vectorize_(false), preferLibs_(false) {}

    ForProperty withParallel(const ParallelScope &parallel) {
        auto ret = *this;
        ret.parallel_ = parallel;
        return ret;
    }
    ForProperty withUnroll(bool unroll = true) {
        auto ret = *this;
        ret.unroll_ = unroll;
        return ret;
    }
    ForProperty withVectorize(bool vectorize = true) {
        auto ret = *this;
        ret.vectorize_ = vectorize;
        return ret;
    }
    ForProperty withNoDeps(const std::vector<std::string> &noDeps) {
        auto ret = *this;
        ret.noDeps_ = noDeps;
        return ret;
    }
    ForProperty withPreferLibs(bool preferLibs = true) {
        auto ret = *this;
        ret.preferLibs_ = preferLibs;
        return ret;
    }
};

class ForNode : public StmtNode {
  public:
    std::string iter_;

    // We also record len_ because it is used in may passes. If we computes len_
    // every time and call simplifyPass to propagate the constants, it is very
    // time consuming
    SubTree<ExprNode> begin_, end_, step_, len_;
    ForProperty property_;
    SubTree<StmtNode> body_;

    void compHash() override;

    DEFINE_NODE_TRAIT(For);
};
typedef Ref<ForNode> For;
#define makeFor(...) makeNode(For, __VA_ARGS__)
template <class Tbegin, class Tend, class Tstep, class Tlen, class Tbody>
Stmt _makeFor(const ID &id, const std::string &iter, Tbegin &&begin, Tend &&end,
              Tstep &&step, Tlen &&len, const ForProperty &property,
              Tbody &&body) {
    For f = For::make();
    f->setId(id);
    f->iter_ = iter;
    f->begin_ = std::forward<Tbegin>(begin);
    f->end_ = std::forward<Tend>(end);
    f->step_ = std::forward<Tstep>(step);
    f->len_ = std::forward<Tlen>(len);
    f->property_ = property;
    f->body_ = std::forward<Tbody>(body);
    return f;
}

class IfNode : public StmtNode {
  public:
    SubTree<ExprNode> cond_;
    SubTree<StmtNode> thenCase_;
    SubTree<StmtNode, NullPolicy::Nullable> elseCase_;

    void compHash() override;

    DEFINE_NODE_TRAIT(If);
};
typedef Ref<IfNode> If;
#define makeIf(...) makeNode(If, __VA_ARGS__)
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt _makeIf(const ID &id, Tcond &&cond, Tthen &&thenCase,
             Telse &&elseCase = nullptr) {
    If i = If::make();
    i->setId(id);
    i->cond_ = std::forward<Tcond>(cond);
    i->thenCase_ = std::forward<Tthen>(thenCase);
    i->elseCase_ = std::forward<Telse>(elseCase);
    return i;
}

/**
 * Assert a condition shall hold in the body
 *
 * Please note that Assert is different from Assume:
 *
 * - An Assert node is written by users to provide EXTRA conditions
 * - An Assert node will be translated to a runtime assertion
 * - Redundant assertions will be removed in simplifying passes
 * - An Assert node is checked in debug/match_ast
 */
class AssertNode : public StmtNode {
  public:
    SubTree<ExprNode> cond_;
    SubTree<StmtNode> body_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Assert);
};
typedef Ref<AssertNode> Assert;
#define makeAssert(...) makeNode(Assert, __VA_ARGS__)
template <class Tcond, class Tbody>
Stmt _makeAssert(const ID &id, Tcond &&cond, Tbody &&body) {
    Assert a = Assert::make();
    a->setId(id);
    a->cond_ = std::forward<Tcond>(cond);
    a->body_ = std::forward<Tbody>(body);
    return a;
}

/**
 * Assume a condition will hold in the body
 *
 * Please note that Assume is different from Assert
 *
 * - An Assume node is introduced by some passes to help the compiler recognize
 * some implicit conditions, and cannot be written by users
 * - An Assume node will not introduce runtime overhead
 * - Assume nodes are inserted and removed freely by passes
 * - An Assert node is NOT checked in debug/match_ast
 */
class AssumeNode : public StmtNode {
  public:
    SubTree<ExprNode> cond_;
    SubTree<StmtNode> body_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Assume);
};
typedef Ref<AssumeNode> Assume;
#define makeAssume(...) makeNode(Assume, __VA_ARGS__)
template <class Tcond, class Tbody>
Stmt _makeAssume(const ID &id, Tcond &&cond, Tbody &&body) {
    Assume a = Assume::make();
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
    void compHash() override;
    DEFINE_NODE_TRAIT(Eval);
};
typedef Ref<EvalNode> Eval;
#define makeEval(...) makeNode(Eval, __VA_ARGS__)
template <class T> Stmt _makeEval(const ID &id, T &&expr) {
    Eval e = Eval::make();
    e->setId(id);
    e->expr_ = std::forward<T>(expr);
    return e;
}

/**
 * External call to a batched GEMM
 */
class MatMulNode : public StmtNode {
  public:
    // c_ = alpha_ * a_ * b_ + beta_ * c_
    // a_ is an m_ * k_ matrix
    // b_ is a k_ * n_ matrix
    // c_ is an m_ * n_ matrix
    SubTree<ExprNode> a_, b_, c_, alpha_, beta_, m_, k_, n_, lda_, ldb_, ldc_,
        stridea_, strideb_, stridec_, batchSize_;
    bool aIsRowMajor_, bIsRowMajor_, cIsRowMajor_;
    SubTree<StmtNode>
        equivalent_; // Equivalent loop statements, to help dependency analysis
    void compHash() override;
    DEFINE_NODE_TRAIT(MatMul);
};
typedef Ref<MatMulNode> MatMul;
#define makeMatMul(...) makeNode(MatMul, __VA_ARGS__)
inline Stmt _makeMatMul(const ID &id, const Expr &a, const Expr &b,
                        const Expr &c, const Expr &alpha, const Expr &beta,
                        const Expr &m, const Expr &k, const Expr &n,
                        const Expr &lda, const Expr &ldb, const Expr &ldc,
                        const Expr &stridea, const Expr &strideb,
                        const Expr &stridec, const Expr &batchSize,
                        bool aIsRowMajor, bool bIsRowMajor, bool cIsRowMajor,
                        const Stmt &equivalent) {
    MatMul s = MatMul::make();
    s->setId(id);
    s->a_ = a;
    s->b_ = b;
    s->c_ = c;
    s->alpha_ = alpha;
    s->beta_ = beta;
    s->m_ = m;
    s->k_ = k;
    s->n_ = n;
    s->lda_ = lda;
    s->ldb_ = ldb;
    s->ldc_ = ldc;
    s->stridea_ = stridea;
    s->strideb_ = strideb;
    s->stridec_ = stridec;
    s->batchSize_ = batchSize;
    s->aIsRowMajor_ = aIsRowMajor;
    s->bIsRowMajor_ = bIsRowMajor;
    s->cIsRowMajor_ = cIsRowMajor;
    s->equivalent_ = equivalent;
    return s;
}

Expr neutralVal(DataType dtype, ReduceOp op);

} // namespace ir

#endif // STMT_H
