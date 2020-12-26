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

    virtual Stmt operator()(const Stmt &op) final;
    virtual Expr operator()(const Expr &op) final;

  protected:
    virtual Stmt visit(const Any &op) { return makeAny(); }

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
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto &&expr = (*this)(op->expr_);
        auto ret = makeStore(op->var_, std::move(indices), std::move(expr));

        std::vector<std::vector<Expr>> info_dep_rw = op->info_dep_rw_;
        std::vector<std::vector<Expr>> info_dep_ww = op->info_dep_ww_;
        for (auto &dep : info_dep_rw) {
            for (auto &expr : dep) {
                expr = (*this)(expr);
            }
        }
        for (auto &dep : info_dep_ww) {
            for (auto &expr : dep) {
                expr = (*this)(expr);
            }
        }
        ret.as<StoreNode>()->info_dep_rw_ = std::move(info_dep_rw);
        ret.as<StoreNode>()->info_dep_ww_ = std::move(info_dep_ww);
        return ret;
    }

    virtual Expr visit(const Load &op) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto ret = makeLoad(op->var_, std::move(indices));

        std::vector<std::vector<Expr>> info_dep_rw = op->info_dep_rw_;
        for (auto &dep : info_dep_rw) {
            for (auto &expr : dep) {
                expr = (*this)(expr);
            }
        }
        ret.as<LoadNode>()->info_dep_rw_ = std::move(info_dep_rw);
        return ret;
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

    virtual Expr visit(const Not &op) { return makeNot((*this)(op->expr_)); }

    virtual Stmt visit(const For &op) {
        return makeFor(op->iter_, (*this)(op->begin_), (*this)(op->end_),
                       (*this)(op->body_), op->id_);
    }

    virtual Stmt visit(const If &op) {
        auto ret =
            makeIf((*this)(op->cond_), (*this)(op->thenCase_),
                   op->elseCase_.isValid() ? (*this)(op->elseCase_) : nullptr);
        if (op->info_equival_cond_.isValid()) {
            ret.as<IfNode>()->info_equival_cond_ =
                (*this)(op->info_equival_cond_);
        }
        return ret;
    }
};

/**
 * Make there will be no shared node in a AST
 *
 * If any pass uses node addresses as keys, perform Disambiguous first
 */
class Disambiguous : public Mutator {
    // Do nothing
};

} // namespace ir

#endif // MUTATOR_H

