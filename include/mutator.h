#ifndef FREE_TENSOR_MUTATOR_H
#define FREE_TENSOR_MUTATOR_H

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <stmt.h>

namespace freetensor {

class Mutator {
  public:
    typedef Expr ExprRetType;
    typedef Stmt StmtRetType;

    virtual ~Mutator() {}

    virtual Stmt operator()(const Stmt &op) final;
    virtual Expr operator()(const Expr &op) final;

  protected:
    // NOTE: Do NOT std::move from the original op! The original op may be
    // duplicated around the AST!

    /* Additional hook for any expressions
     *
     * Cautious when one visitor B inherits another visitor A, the calling order
     * is B::visitExpr -> A::visitExpr -> B::visit -> A::visit
     */
    virtual Expr visitExpr(const Expr &op);

    /* Additional hook for any statements
     *
     * Cautious when one visitor B inherits another visitor A, the calling order
     * is B::visitStmt -> A::visitStmt -> B::visit -> A::visit
     */
    virtual Stmt visitStmt(const Stmt &op);

    virtual Stmt visit(const Any &op) { return makeAny(op->debugBlame()); }

    virtual Stmt visit(const StmtSeq &op) {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
        }
        return makeStmtSeq(std::move(stmts), op->metadata(), op->id(),
                           op->debugBlame());
    }

    virtual Stmt visit(const VarDef &op) {
        std::vector<Expr> shape;
        shape.reserve(op->buffer_->tensor()->shape().size());
        for (auto &&dim : op->buffer_->tensor()->shape()) {
            shape.emplace_back((*this)(dim));
        }
        Ref<Tensor> t =
            makeTensor(std::move(shape), op->buffer_->tensor()->dtype());
        Ref<Buffer> b = makeBuffer(std::move(t), op->buffer_->atype(),
                                   op->buffer_->mtype());
        return makeVarDef(op->name_, std::move(b), op->viewOf_,
                          (*this)(op->body_), op->pinned_, op->metadata(),
                          op->id(), op->debugBlame());
    }

    virtual Expr visit(const Var &op) {
        return makeVar(op->name_, op->debugBlame());
    }

    virtual Stmt visit(const Store &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeStore(op->var_, std::move(indices), std::move(expr),
                         op->metadata(), op->id(), op->debugBlame());
    }

    virtual Stmt visit(const Alloc &op) {
        return makeAlloc(op->var_, op->metadata(), op->id(), op->debugBlame());
    }

    virtual Stmt visit(const Free &op) {
        return makeFree(op->var_, op->metadata(), op->id(), op->debugBlame());
    }

    virtual Expr visit(const Load &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        return makeLoad(op->var_, std::move(indices), op->loadType_,
                        op->debugBlame());
    }

    virtual Stmt visit(const ReduceTo &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeReduceTo(op->var_, std::move(indices), op->op_,
                            std::move(expr), op->sync_, op->metadata(),
                            op->id(), op->debugBlame());
    }

    virtual Expr visit(const AnyExpr &op) {
        return makeAnyExpr(op->debugBlame());
    }

    virtual Expr visit(const IntConst &op) {
        return makeIntConst(op->val_, op->debugBlame());
    }

    virtual Expr visit(const FloatConst &op) {
        return makeFloatConst(op->val_, op->debugBlame());
    }

    virtual Expr visit(const BoolConst &op) {
        return makeBoolConst(op->val_, op->debugBlame());
    }

    virtual Expr visit(const Add &op) {
        return makeAdd((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const Sub &op) {
        return makeSub((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const Mul &op) {
        return makeMul((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const RealDiv &op) {
        return makeRealDiv((*this)(op->lhs_), (*this)(op->rhs_),
                           op->debugBlame());
    }

    virtual Expr visit(const FloorDiv &op) {
        return makeFloorDiv((*this)(op->lhs_), (*this)(op->rhs_),
                            op->debugBlame());
    }

    virtual Expr visit(const CeilDiv &op) {
        return makeCeilDiv((*this)(op->lhs_), (*this)(op->rhs_),
                           op->debugBlame());
    }

    virtual Expr visit(const RoundTowards0Div &op) {
        return makeRoundTowards0Div((*this)(op->lhs_), (*this)(op->rhs_),
                                    op->debugBlame());
    }

    virtual Expr visit(const Mod &op) {
        return makeMod((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const Remainder &op) {
        return makeRemainder((*this)(op->lhs_), (*this)(op->rhs_),
                             op->debugBlame());
    }

    virtual Expr visit(const Min &op) {
        return makeMin((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const Max &op) {
        return makeMax((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const LT &op) {
        return makeLT((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const LE &op) {
        return makeLE((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const GT &op) {
        return makeGT((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const GE &op) {
        return makeGE((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const EQ &op) {
        return makeEQ((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const NE &op) {
        return makeNE((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const LAnd &op) {
        return makeLAnd((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const LOr &op) {
        return makeLOr((*this)(op->lhs_), (*this)(op->rhs_), op->debugBlame());
    }

    virtual Expr visit(const LNot &op) {
        return makeLNot((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Sqrt &op) {
        return makeSqrt((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Exp &op) {
        return makeExp((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Ln &op) {
        return makeLn((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Square &op) {
        return makeSquare((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Sigmoid &op) {
        return makeSigmoid((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Sin &op) {
        return makeSin((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Cos &op) {
        return makeCos((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Tan &op) {
        return makeTan((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Tanh &op) {
        return makeTanh((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Abs &op) {
        return makeAbs((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Floor &op) {
        return makeFloor((*this)(op->expr_), op->debugBlame());
    }

    virtual Expr visit(const Ceil &op) {
        return makeCeil((*this)(op->expr_), op->debugBlame());
    }

    virtual Stmt visit(const For &op) {
        auto begin = (*this)(op->begin_);
        auto end = (*this)(op->end_);
        auto step = (*this)(op->step_);
        auto len = (*this)(op->len_);
        auto property = Ref<ForProperty>::make()
                            ->withParallel(op->property_->parallel_)
                            ->withUnroll(op->property_->unroll_)
                            ->withVectorize(op->property_->vectorize_)
                            ->withNoDeps(op->property_->noDeps_)
                            ->withPreferLibs(op->property_->preferLibs_);
        property->reductions_.reserve(op->property_->reductions_.size());
        for (auto &&r : op->property_->reductions_) {
            std::vector<Expr> begins, ends;
            begins.reserve(r->begins_.size());
            ends.reserve(r->ends_.size());
            for (auto &&item : r->begins_) {
                begins.emplace_back((*this)(item));
            }
            for (auto &&item : r->ends_) {
                ends.emplace_back((*this)(item));
            }
            property->reductions_.emplace_back(makeReductionItem(
                r->op_, r->var_, std::move(begins), std::move(ends)));
        }
        auto body = (*this)(op->body_);
        return makeFor(op->iter_, std::move(begin), std::move(end),
                       std::move(step), std::move(len), std::move(property),
                       std::move(body), op->metadata(), op->id(),
                       op->debugBlame());
    }

    virtual Stmt visit(const If &op) {
        auto cond = (*this)(op->cond_);
        auto thenCase = (*this)(op->thenCase_); // Visit then BEFORE else!
        auto elseCase =
            op->elseCase_.isValid() ? (*this)(op->elseCase_) : nullptr;
        return makeIf(std::move(cond), std::move(thenCase), std::move(elseCase),
                      op->metadata(), op->id(), op->debugBlame());
    }

    virtual Stmt visit(const Assert &op) {
        return makeAssert((*this)(op->cond_), (*this)(op->body_),
                          op->metadata(), op->id(), op->debugBlame());
    }

    virtual Stmt visit(const Assume &op) {
        return makeAssume((*this)(op->cond_), (*this)(op->body_),
                          op->metadata(), op->id(), op->debugBlame());
    }

    virtual Expr visit(const IfExpr &op) {
        return makeIfExpr((*this)(op->cond_), (*this)(op->thenCase_),
                          (*this)(op->elseCase_), op->debugBlame());
    }

    virtual Expr visit(const Cast &op) {
        return makeCast((*this)(op->expr_), op->destType_, op->debugBlame());
    }

    virtual Expr visit(const Intrinsic &op) {
        std::vector<Expr> params;
        params.reserve(op->params_.size());
        for (auto &&param : op->params_) {
            params.emplace_back((*this)(param));
        }
        return makeIntrinsic(op->format_, std::move(params), op->retType_,
                             op->hasSideEffect_, op->debugBlame());
    }

    virtual Stmt visit(const Eval &op) {
        return makeEval((*this)(op->expr_), op->metadata(), op->id(),
                        op->debugBlame());
    }

    virtual Stmt visit(const MatMul &op) {
        return makeMatMul(
            (*this)(op->a_), (*this)(op->b_), (*this)(op->c_),
            (*this)(op->alpha_), (*this)(op->beta_), (*this)(op->m_),
            (*this)(op->k_), (*this)(op->n_), (*this)(op->lda_),
            (*this)(op->ldb_), (*this)(op->ldc_), (*this)(op->stridea_),
            (*this)(op->strideb_), (*this)(op->stridec_),
            (*this)(op->batchSize_), op->aIsRowMajor_, op->bIsRowMajor_,
            op->cIsRowMajor_, (*this)(op->equivalent_), op->metadata(),
            op->id(), op->debugBlame());
    }

    virtual Stmt visit(const MarkVersion &op) {
        return makeMarkVersion(op->tapeName_, op->var_, op->metadata(),
                               op->id(), op->debugBlame());
    }

    virtual Expr visit(const LoadAtVersion &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        return makeLoadAtVersion(op->tapeName_, std::move(indices),
                                 op->loadType_, op->debugBlame());
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_MUTATOR_H
