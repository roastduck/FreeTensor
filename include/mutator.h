#ifndef MUTATOR_H
#define MUTATOR_H

#include <functional>

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

class Mutator {
  public:
    virtual ~Mutator() {}

    virtual Stmt operator()(const Stmt &op) final;
    virtual Expr operator()(const Expr &op) final;

  protected:
    // NOTE: Do NOT std::move from the original op! The original op may be
    // duplicated around the AST!

    // Additional hook for any expressions
    virtual Expr visitExpr(const Expr &op,
                           const std::function<Expr(const Expr &)> &visitNode) {
        return visitNode(op);
    }

    // Additional hook for any statements
    virtual Stmt visitStmt(const Stmt &op,
                           const std::function<Stmt(const Stmt &)> &visitNode) {
        return visitNode(op);
    }

    virtual Stmt visit(const Any &op) { return makeAny(); }

    virtual Stmt visit(const StmtSeq &op) {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
        }
        return makeStmtSeq(op->id(), std::move(stmts));
    }

    virtual Stmt visit(const VarDef &op) {
        std::vector<Expr> shape;
        shape.reserve(op->buffer_->tensor().shape().size());
        for (auto &&dim : op->buffer_->tensor().shape()) {
            shape.emplace_back((*this)(dim));
        }
        Tensor t(std::move(shape), op->buffer_->tensor().dtype());
        Buffer b(std::move(t), op->buffer_->atype(), op->buffer_->mtype());
        return makeVarDef(op->id(), op->name_, std::move(b),
                          (*this)(op->body_));
    }

    virtual Expr visit(const Var &op) { return makeVar(op->name_); }

    virtual Stmt visit(const Store &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeStore(op->id(), op->var_, std::move(indices),
                         std::move(expr));
    }

    virtual Expr visit(const Load &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        return makeLoad(op->var_, std::move(indices));
    }

    virtual Stmt visit(const ReduceTo &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeReduceTo(op->id(), op->var_, std::move(indices), op->op_,
                            std::move(expr), op->atomic_);
    }

    virtual Expr visit(const AnyExpr &op) { return makeAnyExpr(); }

    virtual Expr visit(const IntConst &op) { return makeIntConst(op->val_); }

    virtual Expr visit(const FloatConst &op) {
        return makeFloatConst(op->val_);
    }

    virtual Expr visit(const BoolConst &op) { return makeBoolConst(op->val_); }

    virtual Expr visit(const Add &op) {
        return makeAdd((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Sub &op) {
        return makeSub((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Mul &op) {
        return makeMul((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const RealDiv &op) {
        return makeRealDiv((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const FloorDiv &op) {
        return makeFloorDiv((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const CeilDiv &op) {
        return makeCeilDiv((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const RoundTowards0Div &op) {
        return makeRoundTowards0Div((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Mod &op) {
        return makeMod((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Min &op) {
        auto ret = makeMin((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const Max &op) {
        auto ret = makeMax((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const LT &op) {
        auto ret = makeLT((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const LE &op) {
        auto ret = makeLE((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const GT &op) {
        auto ret = makeGT((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const GE &op) {
        auto ret = makeGE((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const EQ &op) {
        auto ret = makeEQ((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const NE &op) {
        auto ret = makeNE((*this)(op->lhs_), (*this)(op->rhs_));
        return ret;
    }

    virtual Expr visit(const LAnd &op) {
        return makeLAnd((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const LOr &op) {
        return makeLOr((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const LNot &op) { return makeLNot((*this)(op->expr_)); }

    virtual Stmt visit(const For &op) {
        auto ret =
            makeFor(op->id(), op->iter_, (*this)(op->begin_), (*this)(op->end_),
                    op->parallel_, (*this)(op->body_), op->unroll_);
        if (op->infoLen_.isValid()) {
            ret.as<ForNode>()->infoLen_ = (*this)(op->infoLen_);
        }
        return ret;
    }

    virtual Stmt visit(const If &op) {
        auto ret =
            makeIf(op->id(), (*this)(op->cond_), (*this)(op->thenCase_),
                   op->elseCase_.isValid() ? (*this)(op->elseCase_) : nullptr);
        if (op->infoNotCond_.isValid()) {
            ret.as<IfNode>()->infoNotCond_ = (*this)(op->infoNotCond_);
        }
        return ret;
    }

    virtual Stmt visit(const Assert &op) {
        return makeAssert(op->id(), (*this)(op->cond_), (*this)(op->body_));
    }

    virtual Expr visit(const Intrinsic &op) {
        std::vector<Expr> params;
        params.reserve(op->params_.size());
        for (auto &&param : op->params_) {
            params.emplace_back((*this)(param));
        }
        return makeIntrinsic(op->format_, std::move(params));
    }

    virtual Stmt visit(const Eval &op) {
        return makeEval(op->id(), (*this)(op->expr_));
    }
};

} // namespace ir

#endif // MUTATOR_H

