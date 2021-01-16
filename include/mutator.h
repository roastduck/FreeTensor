#ifndef MUTATOR_H
#define MUTATOR_H

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

class Mutator {
  public:
    virtual ~Mutator() {}

    virtual Stmt operator()(const Stmt &op);
    virtual Expr operator()(const Expr &op);

  protected:
    // NOTE: Do NOT std::move from the original op! The original op may be
    // duplicated around the AST!

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
        Buffer b(std::move(t), op->buffer_->atype());
        auto ret =
            makeVarDef(op->id(), op->name_, std::move(b), (*this)(op->body_));
        if (op->info_acc_lower_.isValid()) {
            auto indices = ret.as<VarDefNode>()->info_acc_lower_ =
                Ref<std::vector<Expr>>::make();
            indices->reserve(op->info_acc_lower_->size());
            for (auto &&index : *op->info_acc_lower_) {
                indices->emplace_back((*this)(index));
            }
        }
        if (op->info_acc_len_.isValid()) {
            auto indices = ret.as<VarDefNode>()->info_acc_len_ =
                Ref<std::vector<Expr>>::make();
            indices->reserve(op->info_acc_len_->size());
            for (auto &&index : *op->info_acc_len_) {
                indices->emplace_back((*this)(index));
            }
        }
        return ret;
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

    virtual Stmt visit(const AddTo &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        return makeAddTo(op->id(), op->var_, std::move(indices),
                         std::move(expr));
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

    virtual Expr visit(const Min &op) {
        auto ret = makeMin((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<LTNode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const Max &op) {
        auto ret = makeMax((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<LTNode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const LT &op) {
        auto ret = makeLT((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<LTNode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const LE &op) {
        auto ret = makeLE((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<LENode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const GT &op) {
        auto ret = makeGT((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<GTNode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const GE &op) {
        auto ret = makeGE((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<GENode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const EQ &op) {
        auto ret = makeEQ((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<EQNode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const NE &op) {
        auto ret = makeNE((*this)(op->lhs_), (*this)(op->rhs_));
        if (op->info_norm_form_.isValid()) {
            ret.as<NENode>()->info_norm_form_ = (*this)(op->info_norm_form_);
        }
        return ret;
    }

    virtual Expr visit(const Not &op) { return makeNot((*this)(op->expr_)); }

    virtual Stmt visit(const For &op) {
        auto ret =
            makeFor(op->id(), op->iter_, (*this)(op->begin_), (*this)(op->end_),
                    op->parallel_, (*this)(op->body_));
        if (op->info_len_.isValid()) {
            ret.as<ForNode>()->info_len_ = (*this)(op->info_len_);
        }
        if (op->info_max_begin_.isValid()) {
            ret.as<ForNode>()->info_max_begin_ = (*this)(op->info_max_begin_);
        }
        if (op->info_min_end_.isValid()) {
            ret.as<ForNode>()->info_min_end_ = (*this)(op->info_min_end_);
        }
        return ret;
    }

    virtual Stmt visit(const If &op) {
        return makeIf(op->id(), (*this)(op->cond_), (*this)(op->thenCase_),
                      op->elseCase_.isValid() ? (*this)(op->elseCase_)
                                              : nullptr);
    }

    virtual Stmt visit(const Assert &op) {
        return makeAssert(op->id(), (*this)(op->cond_), (*this)(op->body_));
    }
};

} // namespace ir

#endif // MUTATOR_H

