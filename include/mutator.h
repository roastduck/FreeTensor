#ifndef MUTATOR_H
#define MUTATOR_H

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
    virtual Stmt visit(const StmtSeq &op) {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
        }
        return makeStmtSeq(std::move(stmts));
    }

    virtual Stmt visit(const VarDef &op) {
        std::vector<Expr> shape;
        shape.reserve(op->buffer_->tensor().shape().size());
        for (auto &&dim : op->buffer_->tensor().shape()) {
            shape.emplace_back((*this)(dim));
        }
        Tensor t(std::move(shape), op->buffer_->tensor().dtype());
        Buffer b(t, op->buffer_->atype());
        return makeVarDef(op->name_, std::move(b), (*this)(op->body_));
    }

    virtual Expr visit(const Var &op) { return makeVar(op->name_); }

    virtual Stmt visit(const Store &op) {
        auto &&var = (*this)(op->var_);
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeStore(std::move(var), std::move(indices), std::move(expr));
    }

    virtual Expr visit(const Load &op) {
        auto &&var = (*this)(op->var_);
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        return makeLoad(std::move(var), std::move(indices));
    }

    virtual Expr visit(const IntConst &op) { return makeIntConst(op->val_); }

    virtual Expr visit(const FloatConst &op) {
        return makeFloatConst(op->val_);
    }

    virtual Expr visit(const Add &op) {
        return makeAdd((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Sub &op) {
        return makeSub((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Mul &op) {
        return makeMul((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Div &op) {
        return makeDiv((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const Mod &op) {
        return makeMod((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const LT &op) {
        return makeLT((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const LE &op) {
        return makeLE((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const GT &op) {
        return makeGT((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const GE &op) {
        return makeGE((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const EQ &op) {
        return makeEQ((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Expr visit(const NE &op) {
        return makeNE((*this)(op->lhs_), (*this)(op->rhs_));
    }

    virtual Stmt visit(const For &op) {
        return makeFor(op->iter_, (*this)(op->begin_), (*this)(op->end_),
                       (*this)(op->body_));
    }

    virtual Stmt visit(const If &op) {
        return makeIf((*this)(op->cond_), (*this)(op->thenCase_),
                      op->elseCase_.isValid() ? (*this)(op->elseCase_)
                                              : nullptr);
    }
};

} // namespace ir

#endif // MUTATOR_H

