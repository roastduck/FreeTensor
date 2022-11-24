#ifndef FREE_TENSOR_STMT_H
#define FREE_TENSOR_STMT_H

#include <string>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <for_property.h>
#include <reduce_op.h>

namespace freetensor {

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

/**
 * A sequence of any number of statements, or a so-called statement block in
 * some languages
 *
 * Tips: turning a statement into an empty StmtSeq is a handful way to delete
 * the statment in a Mutator
 */
class StmtSeqNode : public StmtNode {
  public:
    SubTreeList<StmtNode> stmts_ = ChildOf{this};
    std::vector<Stmt> children() const override { return stmts_; }
    void compHash() override;
    DEFINE_NODE_TRAIT(StmtSeq);
};
typedef Ref<StmtSeqNode> StmtSeq;
#define makeStmtSeq(...) makeNode(StmtSeq, __VA_ARGS__)
template <class Tstmts>
Stmt _makeStmtSeq(Tstmts &&stmts, const Metadata &metadata = nullptr,
                  const ID &id = {}) {
    StmtSeq s = StmtSeq::make();
    s->metadata() = metadata;
    s->setId(id);
    s->stmts_ = std::forward<Tstmts>(stmts);
    return s;
}
inline Stmt _makeStmtSeq(std::initializer_list<Stmt> stmts,
                         const Metadata &metadata = nullptr,
                         const ID &id = {}) {
    StmtSeq s = StmtSeq::make();
    s->metadata() = metadata;
    s->setId(id);
    s->stmts_ = stmts;
    return s;
}

/**
 * Define a (tensor) variable
 *
 * The scope of the variable is `body_`, which means scopes in FreeTensor
 * follows FILO (stack) rule
 *
 * It is NOT allowed to modify the shape of a `VarDef` inside its body. This
 * rule is enforced in the frontend (see the checkings on `borrowed_vardefs` in
 * `VarRef` in `expr.py`) and each pass should keep this rule
 */
class VarDefNode : public StmtNode {
  public:
    std::string name_;
    SubTree<Buffer> buffer_ = ChildOf{this};

    /**
     * This `VarDef` node can be a view of another `VarDef` node, represented by
     * the name of its source `VarDef`
     *
     * Simultaneously access of a `VarDef` and the `VarDef` it views is ALWAYS
     * treated as dependences
     *
     * This is useful when we want to alter the shape of an `VarDef` with
     * non-`cache` access type, whose shape must be kept during I/O or in an
     * internal allocation
     */
    std::optional<std::string> viewOf_;

    SubTree<StmtNode> body_ = ChildOf{this};
    bool pinned_; /// If pinned, SinkVar and ShrinkVar will not alter this node
    std::vector<Stmt> children() const override { return {body_}; }
    void compHash() override;
    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
#define makeVarDef(...) makeNode(VarDef, __VA_ARGS__)
template <class Tbuffer, class Tbody>
Stmt _makeVarDef(const std::string &name, Tbuffer &&buffer,
                 const std::optional<std::string> &viewOf, Tbody &&body,
                 bool pinned, const Metadata &metadata = nullptr,
                 const ID &id = {}) {
    ASSERT(!name.empty());
    VarDef d = VarDef::make();
    d->metadata() = metadata;
    d->setId(id);
    d->name_ = name;
    d->buffer_ = SubTree<Buffer>(buffer);
    d->viewOf_ = viewOf;
    d->body_ = std::forward<Tbody>(body);
    d->pinned_ = pinned;
    return d;
}

/**
 * Assignment
 *
 * Semantics: `var_[indices_] = expr_`
 */
class StoreNode : public StmtNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_ = ChildOf{this};
    SubTree<ExprNode> expr_ = ChildOf{this};
    void compHash() override;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
#define makeStore(...) makeNode(Store, __VA_ARGS__)
template <class Tindices, class Texpr>
Stmt _makeStore(const std::string &var, Tindices &&indices, Texpr &&expr,
                const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!var.empty());
    Store s = Store::make();
    s->metadata() = metadata;
    s->setId(id);
    s->var_ = var;
    s->indices_ = std::forward<Tindices>(indices);
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}
template <class Texpr>
Stmt _makeStore(const std::string &var, const std::vector<Expr> &indices,
                Texpr &&expr, const Metadata &metadata = nullptr,
                const ID &id = {}) {
    ASSERT(!var.empty());
    Store s = Store::make();
    s->metadata() = metadata;
    s->setId(id);
    s->var_ = var;
    s->indices_ = indices;
    s->expr_ = std::forward<Texpr>(expr);
    return s;
}

/**
 * Explicit memory allocation
 *
 * Only used for variables with `.../heap` memory types
 */
class AllocNode : public StmtNode {
  public:
    std::string var_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Alloc);
};
typedef Ref<AllocNode> Alloc;
#define makeAlloc(...) makeNode(Alloc, __VA_ARGS__)
inline Stmt _makeAlloc(const std::string &var,
                       const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!var.empty());
    Alloc a = Alloc::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    return a;
}

/**
 * Explicit memory free
 *
 * Only used for variables with `.../heap` memory types
 */
class FreeNode : public StmtNode {
  public:
    std::string var_;
    void compHash() override;
    DEFINE_NODE_TRAIT(Free);
};
typedef Ref<FreeNode> Free;
#define makeFree(...) makeNode(Free, __VA_ARGS__)
inline Stmt _makeFree(const std::string &var,
                      const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!var.empty());
    Free f = Free::make();
    f->metadata() = metadata;
    f->setId(id);
    f->var_ = var;
    return f;
}

/**
 * A variant of `Store` for `+=`, `-=`, `*=`, etc
 *
 * Example: `var_[indices_] += expr_`
 *
 * These operations follow commutative law. Making them the special `ReduceTo`
 * nodes helps analyzing dependences more accurately
 */
class ReduceToNode : public StmtNode {
  public:
    std::string var_;
    SubTreeList<ExprNode> indices_ = ChildOf{this};
    ReduceOp op_;
    SubTree<ExprNode> expr_ = ChildOf{this};
    bool atomic_;
    void compHash() override;
    DEFINE_NODE_TRAIT(ReduceTo)
};
typedef Ref<ReduceToNode> ReduceTo;
#define makeReduceTo(...) makeNode(ReduceTo, __VA_ARGS__)
template <class Tindices, class Texpr>
Stmt _makeReduceTo(const std::string &var, Tindices &&indices, ReduceOp op,
                   Texpr &&expr, bool atomic,
                   const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!var.empty());
    ReduceTo a = ReduceTo::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    a->indices_ = std::forward<Tindices>(indices);
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->atomic_ = atomic;
    return a;
}
template <class Texpr>
Stmt _makeReduceTo(const std::string &var, const std::vector<Expr> &indices,
                   ReduceOp op, Texpr &&expr, bool atomic,
                   const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!var.empty());
    ReduceTo a = ReduceTo::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    a->indices_ = indices;
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->atomic_ = atomic;
    return a;
}

/**
 * For loop in a integral range
 */
class ForNode : public StmtNode {
  public:
    std::string iter_;

    // We also record len_ because it is used in may passes. If we computes len_
    // every time and call simplify to propagate the constants, it is very
    // time consuming
    SubTree<ExprNode> begin_ = ChildOf{this};
    SubTree<ExprNode> end_ = ChildOf{this};
    SubTree<ExprNode> step_ = ChildOf{this};
    SubTree<ExprNode> len_ = ChildOf{this};
    SubTree<ForProperty> property_ = ChildOf{this};
    SubTree<StmtNode> body_ = ChildOf{this};

    bool isCtrlFlow() const override { return true; }
    std::vector<Stmt> children() const override { return {body_}; }

    void compHash() override;

    DEFINE_NODE_TRAIT(For);
};
typedef Ref<ForNode> For;
#define makeFor(...) makeNode(For, __VA_ARGS__)
template <class Tbegin, class Tend, class Tstep, class Tlen, class Tbody,
          class Tproperty>
Stmt _makeFor(const std::string &iter, Tbegin &&begin, Tend &&end, Tstep &&step,
              Tlen &&len, Tproperty &&property, Tbody &&body,
              const Metadata &metadata = nullptr, const ID &id = {}) {
    ASSERT(!iter.empty());
    For f = For::make();
    f->metadata() = metadata;
    f->setId(id);
    f->iter_ = iter;
    f->begin_ = std::forward<Tbegin>(begin);
    f->end_ = std::forward<Tend>(end);
    f->step_ = std::forward<Tstep>(step);
    f->len_ = std::forward<Tlen>(len);
    f->property_ = std::forward<Tproperty>(property);
    f->body_ = std::forward<Tbody>(body);
    return f;
}

/**
 * If-then-else branch
 */
class IfNode : public StmtNode {
  public:
    SubTree<ExprNode> cond_ = ChildOf{this};
    SubTree<StmtNode> thenCase_ = ChildOf{this};
    SubTree<StmtNode, NullPolicy::Nullable> elseCase_ = ChildOf{this};

    bool isCtrlFlow() const override { return true; }
    std::vector<Stmt> children() const override {
        if (elseCase_.isValid()) {
            return {thenCase_, elseCase_};
        } else {
            return {thenCase_};
        }
    }

    void compHash() override;

    DEFINE_NODE_TRAIT(If);
};
typedef Ref<IfNode> If;
#define makeIf(...) makeNode(If, __VA_ARGS__)
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt _makeIf(Tcond &&cond, Tthen &&thenCase, Telse &&elseCase,
             const Metadata &metadata = nullptr, const ID &id = {}) {
    If i = If::make();
    i->metadata() = metadata;
    i->setId(id);
    i->cond_ = std::forward<Tcond>(cond);
    i->thenCase_ = std::forward<Tthen>(thenCase);
    i->elseCase_ = std::forward<Telse>(elseCase);
    return i;
}
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt _makeIf(Tcond &&cond, Tthen &&thenCase, const Metadata &metadata = nullptr,
             const ID &id = {}) {
    return _makeIf(cond, thenCase, nullptr, metadata, id);
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
    SubTree<ExprNode> cond_ = ChildOf{this};
    SubTree<StmtNode> body_ = ChildOf{this};
    bool isCtrlFlow() const override { return true; }
    std::vector<Stmt> children() const override { return {body_}; }
    void compHash() override;
    DEFINE_NODE_TRAIT(Assert);
};
typedef Ref<AssertNode> Assert;
#define makeAssert(...) makeNode(Assert, __VA_ARGS__)
template <class Tcond, class Tbody>
Stmt _makeAssert(Tcond &&cond, Tbody &&body, const Metadata &metadata = nullptr,
                 const ID &id = {}) {
    Assert a = Assert::make();
    a->metadata() = metadata;
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
    SubTree<ExprNode> cond_ = ChildOf{this};
    SubTree<StmtNode> body_ = ChildOf{this};
    std::vector<Stmt> children() const override { return {body_}; }
    void compHash() override;
    DEFINE_NODE_TRAIT(Assume);
};
typedef Ref<AssumeNode> Assume;
#define makeAssume(...) makeNode(Assume, __VA_ARGS__)
template <class Tcond, class Tbody>
Stmt _makeAssume(Tcond &&cond, Tbody &&body, const Metadata &metadata = nullptr,
                 const ID &id = {}) {
    Assume a = Assume::make();
    a->metadata() = metadata;
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
    SubTree<ExprNode> expr_ = ChildOf{this};
    void compHash() override;
    DEFINE_NODE_TRAIT(Eval);
};
typedef Ref<EvalNode> Eval;
#define makeEval(...) makeNode(Eval, __VA_ARGS__)
template <class T>
Stmt _makeEval(T &&expr, const Metadata &metadata = nullptr,
               const ID &id = {}) {
    Eval e = Eval::make();
    e->metadata() = metadata;
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
    SubTree<ExprNode> a_ = ChildOf{this};
    SubTree<ExprNode> b_ = ChildOf{this};
    SubTree<ExprNode> c_ = ChildOf{this};
    SubTree<ExprNode> alpha_ = ChildOf{this};
    SubTree<ExprNode> beta_ = ChildOf{this};
    SubTree<ExprNode> m_ = ChildOf{this};
    SubTree<ExprNode> k_ = ChildOf{this};
    SubTree<ExprNode> n_ = ChildOf{this};
    SubTree<ExprNode> lda_ = ChildOf{this};
    SubTree<ExprNode> ldb_ = ChildOf{this};
    SubTree<ExprNode> ldc_ = ChildOf{this};
    SubTree<ExprNode> stridea_ = ChildOf{this};
    SubTree<ExprNode> strideb_ = ChildOf{this};
    SubTree<ExprNode> stridec_ = ChildOf{this};
    SubTree<ExprNode> batchSize_ = ChildOf{this};
    bool aIsRowMajor_, bIsRowMajor_, cIsRowMajor_;
    SubTree<StmtNode> equivalent_ = ChildOf{
        this}; // Equivalent loop statements, to help dependence analysis
    std::vector<Stmt> children() const override { return {equivalent_}; }
    void compHash() override;
    DEFINE_NODE_TRAIT(MatMul);
};
typedef Ref<MatMulNode> MatMul;
#define makeMatMul(...) makeNode(MatMul, __VA_ARGS__)
inline Stmt _makeMatMul(const Expr &a, const Expr &b, const Expr &c,
                        const Expr &alpha, const Expr &beta, const Expr &m,
                        const Expr &k, const Expr &n, const Expr &lda,
                        const Expr &ldb, const Expr &ldc, const Expr &stridea,
                        const Expr &strideb, const Expr &stridec,
                        const Expr &batchSize, bool aIsRowMajor,
                        bool bIsRowMajor, bool cIsRowMajor,
                        const Stmt &equivalent,
                        const Metadata &metadata = nullptr, const ID &id = {}) {
    MatMul s = MatMul::make();
    s->metadata() = metadata;
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

} // namespace freetensor

#endif // FREE_TENSOR_STMT_H
