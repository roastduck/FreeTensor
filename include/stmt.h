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
inline Stmt
makeAny(std::source_location loc = std::source_location::current()) {
    Any a = Any::make();
    a->setDebugBlame(loc);
    return a;
}

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
template <class Tstmts>
Stmt makeStmtSeq(Tstmts &&stmts, const Metadata &metadata = nullptr,
                 const ID &id = {},
                 std::source_location loc = std::source_location::current()) {
    StmtSeq s = StmtSeq::make();
    s->metadata() = metadata;
    s->setId(id);
    s->stmts_ = std::forward<Tstmts>(stmts);
    s->setDebugBlame(loc);
    return s;
}
inline Stmt
makeStmtSeq(std::initializer_list<Stmt> stmts,
            const Metadata &metadata = nullptr, const ID &id = {},
            std::source_location loc = std::source_location::current()) {
    StmtSeq s = StmtSeq::make();
    s->metadata() = metadata;
    s->setId(id);
    s->stmts_ = stmts;
    s->setDebugBlame(loc);
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
template <class Tbuffer, class Tbody>
Stmt makeVarDef(const std::string &name, Tbuffer &&buffer,
                const std::optional<std::string> &viewOf, Tbody &&body,
                bool pinned, const Metadata &metadata = nullptr,
                const ID &id = {},
                std::source_location loc = std::source_location::current()) {
    ASSERT(!name.empty());
    VarDef d = VarDef::make();
    d->metadata() = metadata;
    d->setId(id);
    d->name_ = name;
    d->buffer_ = std::forward<Tbuffer>(buffer);
    d->viewOf_ = viewOf;
    d->body_ = std::forward<Tbody>(body);
    d->pinned_ = pinned;
    d->setDebugBlame(loc);
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
template <class Tindices, class Texpr>
Stmt makeStore(const std::string &var, Tindices &&indices, Texpr &&expr,
               const Metadata &metadata = nullptr, const ID &id = {},
               std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Store s = Store::make();
    s->metadata() = metadata;
    s->setId(id);
    s->var_ = var;
    s->indices_ = std::forward<Tindices>(indices);
    s->expr_ = std::forward<Texpr>(expr);
    s->setDebugBlame(loc);
    return s;
}
template <class Texpr>
Stmt makeStore(const std::string &var, const std::vector<Expr> &indices,
               Texpr &&expr, const Metadata &metadata = nullptr,
               const ID &id = {},
               std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Store s = Store::make();
    s->metadata() = metadata;
    s->setId(id);
    s->var_ = var;
    s->indices_ = indices;
    s->expr_ = std::forward<Texpr>(expr);
    s->setDebugBlame(loc);
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
inline Stmt
makeAlloc(const std::string &var, const Metadata &metadata = nullptr,
          const ID &id = {},
          std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Alloc a = Alloc::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    a->setDebugBlame(loc);
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
inline Stmt
makeFree(const std::string &var, const Metadata &metadata = nullptr,
         const ID &id = {},
         std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    Free f = Free::make();
    f->metadata() = metadata;
    f->setId(id);
    f->var_ = var;
    f->setDebugBlame(loc);
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

    /// If true, can safely reduce from multiple threads to one location. Prefer
    /// atomic over other synchronizations
    ///
    /// NOTE: Synchronizations only have to be done among `ReduceTo` nodes. No
    /// need to synchronize between a `ReduceTo` and a `Load`, or a `ReduceTo`
    /// and a `Store`, since our schedules prohibits simultaneous `ReduceTo` and
    /// `Load`, or `ReduceTo` and `Store`.
    bool sync_;

    void compHash() override;
    DEFINE_NODE_TRAIT(ReduceTo)
};
typedef Ref<ReduceToNode> ReduceTo;
template <class Tindices, class Texpr>
Stmt makeReduceTo(const std::string &var, Tindices &&indices, ReduceOp op,
                  Texpr &&expr, bool sync, const Metadata &metadata = nullptr,
                  const ID &id = {},
                  std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    ReduceTo a = ReduceTo::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    a->indices_ = std::forward<Tindices>(indices);
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->sync_ = sync;
    a->setDebugBlame(loc);
    return a;
}
template <class Texpr>
Stmt makeReduceTo(const std::string &var, const std::vector<Expr> &indices,
                  ReduceOp op, Texpr &&expr, bool sync,
                  const Metadata &metadata = nullptr, const ID &id = {},
                  std::source_location loc = std::source_location::current()) {
    ASSERT(!var.empty());
    ReduceTo a = ReduceTo::make();
    a->metadata() = metadata;
    a->setId(id);
    a->var_ = var;
    a->indices_ = indices;
    a->op_ = op;
    a->expr_ = std::forward<Texpr>(expr);
    a->sync_ = sync;
    a->setDebugBlame(loc);
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
template <class Tbegin, class Tend, class Tstep, class Tlen, class Tbody,
          class Tproperty>
Stmt makeFor(const std::string &iter, Tbegin &&begin, Tend &&end, Tstep &&step,
             Tlen &&len, Tproperty &&property, Tbody &&body,
             const Metadata &metadata = nullptr, const ID &id = {},
             std::source_location loc = std::source_location::current()) {
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
    f->setDebugBlame(loc);
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
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt makeIf(Tcond &&cond, Tthen &&thenCase, Telse &&elseCase,
            const Metadata &metadata = nullptr, const ID &id = {},
            std::source_location loc = std::source_location::current()) {
    If i = If::make();
    i->metadata() = metadata;
    i->setId(id);
    i->cond_ = std::forward<Tcond>(cond);
    i->thenCase_ = std::forward<Tthen>(thenCase);
    i->elseCase_ = std::forward<Telse>(elseCase);
    i->setDebugBlame(loc);
    return i;
}
template <class Tcond, class Tthen, class Telse = std::nullptr_t>
Stmt makeIf(Tcond &&cond, Tthen &&thenCase, const Metadata &metadata = nullptr,
            const ID &id = {},
            std::source_location loc = std::source_location::current()) {
    return makeIf(cond, thenCase, nullptr, metadata, id, loc);
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
template <class Tcond, class Tbody>
Stmt makeAssert(Tcond &&cond, Tbody &&body, const Metadata &metadata = nullptr,
                const ID &id = {},
                std::source_location loc = std::source_location::current()) {
    Assert a = Assert::make();
    a->metadata() = metadata;
    a->setId(id);
    a->cond_ = std::forward<Tcond>(cond);
    a->body_ = std::forward<Tbody>(body);
    a->setDebugBlame(loc);
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
template <class Tcond, class Tbody>
Stmt makeAssume(Tcond &&cond, Tbody &&body, const Metadata &metadata = nullptr,
                const ID &id = {},
                std::source_location loc = std::source_location::current()) {
    Assume a = Assume::make();
    a->metadata() = metadata;
    a->setId(id);
    a->cond_ = std::forward<Tcond>(cond);
    a->body_ = std::forward<Tbody>(body);
    a->setDebugBlame(loc);
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
template <class T>
Stmt makeEval(T &&expr, const Metadata &metadata = nullptr, const ID &id = {},
              std::source_location loc = std::source_location::current()) {
    Eval e = Eval::make();
    e->metadata() = metadata;
    e->setId(id);
    e->expr_ = std::forward<T>(expr);
    e->setDebugBlame(loc);
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
inline Stmt
makeMatMul(const Expr &a, const Expr &b, const Expr &c, const Expr &alpha,
           const Expr &beta, const Expr &m, const Expr &k, const Expr &n,
           const Expr &lda, const Expr &ldb, const Expr &ldc,
           const Expr &stridea, const Expr &strideb, const Expr &stridec,
           const Expr &batchSize, bool aIsRowMajor, bool bIsRowMajor,
           bool cIsRowMajor, const Stmt &equivalent,
           const Metadata &metadata = nullptr, const ID &id = {},
           std::source_location loc = std::source_location::current()) {
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
    s->setDebugBlame(loc);
    return s;
}

class MarkVersionNode : public StmtNode {
  public:
    std::string tapeName_, var_;
    void compHash() override;
    DEFINE_NODE_TRAIT(MarkVersion);
};
typedef Ref<MarkVersionNode> MarkVersion;
inline Stmt
makeMarkVersion(const std::string &tapeName, const std::string &var,
                const Metadata &metadata = nullptr, const ID &id = {},
                std::source_location loc = std::source_location::current()) {
    MarkVersion s = MarkVersion::make();
    s->metadata() = metadata;
    s->setId(id);
    s->tapeName_ = tapeName;
    s->var_ = var;
    s->setDebugBlame(loc);
    return s;
}

} // namespace freetensor

#endif // FREE_TENSOR_STMT_H
