#ifndef MUTATOR_H
#define MUTATOR_H

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

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

    virtual Stmt visit(const Any &op) { return COPY_DEBUG_INFO(makeAny(), op); }

    virtual Stmt visit(const StmtSeq &op) {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
        }
        return COPY_DEBUG_INFO(makeStmtSeq(op->id(), std::move(stmts)), op);
    }

    virtual Stmt visit(const VarDef &op) {
        std::vector<SubTree<ExprNode>> shape;
        shape.reserve(op->buffer_->tensor().shape().size());
        for (auto &&dim : op->buffer_->tensor().shape()) {
            shape.emplace_back((*this)(dim));
        }
        Tensor t(std::move(shape), op->buffer_->tensor().dtype());
        Buffer b(std::move(t), op->buffer_->atype(), op->buffer_->mtype());
        Expr sizeLim = op->sizeLim_.isValid() ? (*this)(op->sizeLim_) : nullptr;
        return COPY_DEBUG_INFO(makeVarDef(op->id(), op->name_, std::move(b),
                                          std::move(sizeLim),
                                          (*this)(op->body_), op->pinned_),
                               op);
    }

    virtual Expr visit(const Var &op) {
        return COPY_DEBUG_INFO(makeVar(op->name_), op);
    }

    virtual Stmt visit(const Store &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return COPY_DEBUG_INFO(
            makeStore(op->id(), op->var_, std::move(indices), std::move(expr)),
            op);
    }

    virtual Expr visit(const Load &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        return COPY_DEBUG_INFO(makeLoad(op->var_, std::move(indices)), op);
    }

    virtual Stmt visit(const ReduceTo &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return COPY_DEBUG_INFO(makeReduceTo(op->id(), op->var_,
                                            std::move(indices), op->op_,
                                            std::move(expr), op->atomic_),
                               op);
    }

    virtual Expr visit(const AnyExpr &op) {
        return COPY_DEBUG_INFO(makeAnyExpr(), op);
    }

    virtual Expr visit(const IntConst &op) {
        return COPY_DEBUG_INFO(makeIntConst(op->val_), op);
    }

    virtual Expr visit(const FloatConst &op) {
        return COPY_DEBUG_INFO(makeFloatConst(op->val_), op);
    }

    virtual Expr visit(const BoolConst &op) {
        return COPY_DEBUG_INFO(makeBoolConst(op->val_), op);
    }

    virtual Expr visit(const Add &op) {
        return COPY_DEBUG_INFO(makeAdd((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const Sub &op) {
        return COPY_DEBUG_INFO(makeSub((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const Mul &op) {
        return COPY_DEBUG_INFO(makeMul((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const RealDiv &op) {
        return COPY_DEBUG_INFO(
            makeRealDiv((*this)(op->lhs_), (*this)(op->rhs_)), op);
    }

    virtual Expr visit(const FloorDiv &op) {
        return COPY_DEBUG_INFO(
            makeFloorDiv((*this)(op->lhs_), (*this)(op->rhs_)), op);
    }

    virtual Expr visit(const CeilDiv &op) {
        return COPY_DEBUG_INFO(
            makeCeilDiv((*this)(op->lhs_), (*this)(op->rhs_)), op);
    }

    virtual Expr visit(const RoundTowards0Div &op) {
        return COPY_DEBUG_INFO(
            makeRoundTowards0Div((*this)(op->lhs_), (*this)(op->rhs_)), op);
    }

    virtual Expr visit(const Mod &op) {
        return COPY_DEBUG_INFO(makeMod((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const Remainder &op) {
        return COPY_DEBUG_INFO(
            makeRemainder((*this)(op->lhs_), (*this)(op->rhs_)), op);
    }

    virtual Expr visit(const Min &op) {
        return COPY_DEBUG_INFO(makeMin((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const Max &op) {
        return COPY_DEBUG_INFO(makeMax((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const LT &op) {
        return COPY_DEBUG_INFO(makeLT((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const LE &op) {
        return COPY_DEBUG_INFO(makeLE((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const GT &op) {
        return COPY_DEBUG_INFO(makeGT((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const GE &op) {
        return COPY_DEBUG_INFO(makeGE((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const EQ &op) {
        return COPY_DEBUG_INFO(makeEQ((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const NE &op) {
        return COPY_DEBUG_INFO(makeNE((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const LAnd &op) {
        return COPY_DEBUG_INFO(makeLAnd((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const LOr &op) {
        return COPY_DEBUG_INFO(makeLOr((*this)(op->lhs_), (*this)(op->rhs_)),
                               op);
    }

    virtual Expr visit(const LNot &op) {
        return COPY_DEBUG_INFO(makeLNot((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Sqrt &op) {
        return COPY_DEBUG_INFO(makeSqrt((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Exp &op) {
        return COPY_DEBUG_INFO(makeExp((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Square &op) {
        return COPY_DEBUG_INFO(makeSquare((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Sigmoid &op) {
        return COPY_DEBUG_INFO(makeSigmoid((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Tanh &op) {
        return COPY_DEBUG_INFO(makeTanh((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Abs &op) {
        return COPY_DEBUG_INFO(makeAbs((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Floor &op) {
        return COPY_DEBUG_INFO(makeFloor((*this)(op->expr_)), op);
    }

    virtual Expr visit(const Ceil &op) {
        return COPY_DEBUG_INFO(makeCeil((*this)(op->expr_)), op);
    }

    virtual Stmt visit(const For &op) {
        auto ret =
            makeFor(op->id(), op->iter_, (*this)(op->begin_), (*this)(op->end_),
                    (*this)(op->step_), (*this)(op->len_), op->property_,
                    (*this)(op->body_));
        return COPY_DEBUG_INFO(ret, op);
    }

    virtual Stmt visit(const If &op) {
        auto cond = (*this)(op->cond_);
        auto thenCase = (*this)(op->thenCase_); // Visit then BEFORE else!
        auto elseCase =
            op->elseCase_.isValid() ? (*this)(op->elseCase_) : nullptr;
        auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                          std::move(elseCase));
        return COPY_DEBUG_INFO(ret, op);
    }

    virtual Stmt visit(const Assert &op) {
        return COPY_DEBUG_INFO(
            makeAssert(op->id(), (*this)(op->cond_), (*this)(op->body_)), op);
    }

    virtual Expr visit(const IfExpr &op) {
        return COPY_DEBUG_INFO(makeIfExpr((*this)(op->cond_),
                                          (*this)(op->thenCase_),
                                          (*this)(op->elseCase_)),
                               op);
    }

    virtual Expr visit(const Cast &op) {
        return COPY_DEBUG_INFO(makeCast((*this)(op->expr_), op->dtype_), op);
    }

    virtual Expr visit(const Intrinsic &op) {
        std::vector<Expr> params;
        params.reserve(op->params_.size());
        for (auto &&param : op->params_) {
            params.emplace_back((*this)(param));
        }
        return COPY_DEBUG_INFO(
            makeIntrinsic(op->format_, std::move(params), op->retType_), op);
    }

    virtual Stmt visit(const Eval &op) {
        return COPY_DEBUG_INFO(makeEval(op->id(), (*this)(op->expr_)), op);
    }

    virtual Stmt visit(const MatMul &op) {
        return COPY_DEBUG_INFO(
            makeMatMul(op->id(), (*this)(op->a_), (*this)(op->b_),
                       (*this)(op->c_), (*this)(op->alpha_), (*this)(op->beta_),
                       (*this)(op->m_), (*this)(op->k_), (*this)(op->n_),
                       (*this)(op->lda_), (*this)(op->ldb_), (*this)(op->ldc_),
                       (*this)(op->stridea_), (*this)(op->strideb_),
                       (*this)(op->stridec_), (*this)(op->batchSize_),
                       op->aIsRowMajor_, op->bIsRowMajor_, op->cIsRowMajor_,
                       (*this)(op->equivalent_)),
            op);
    }
};

} // namespace ir

#endif // MUTATOR_H
