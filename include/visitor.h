#ifndef FREE_TENSOR_VISITOR_H
#define FREE_TENSOR_VISITOR_H

#include <debug.h>
#include <expr.h>
#include <func.h>
#include <stmt.h>

namespace freetensor {

class Visitor {
  public:
    typedef void ExprRetType;
    typedef void StmtRetType;

    virtual ~Visitor() {}

    virtual void operator()(const AST &op) final;

  protected:
    /* Additional hook for any expressions
     *
     * Cautious when one visitor B inherits another visitor A, the calling order
     * is B::visitExpr -> A::visitExpr -> B::visit -> A::visit
     */
    virtual void visitExpr(const Expr &op);

    /* Additional hook for any statements
     *
     * Cautious when one visitor B inherits another visitor A, the calling order
     * is B::visitStmt -> A::visitStmt -> B::visit -> A::visit
     */
    virtual void visitStmt(const Stmt &op);

    // Hooks for each node
    virtual void visit(const Any &op) {}
    virtual void visit(const AnyExpr &op) {}

    virtual void visit(const Func &op) { (*this)(op->body_); }

    virtual void visit(const StmtSeq &op) {
        for (auto &&stmt : op->stmts_) {
            (*this)(stmt);
        }
    }

    virtual void visit(const VarDef &op) {
        for (auto &&dim : op->buffer_->tensor()->shape()) {
            (*this)(dim);
        }
        if (op->ioTensor_.isValid()) {
            for (auto &&dim : op->ioTensor_->shape()) {
                (*this)(dim);
            }
        }
        (*this)(op->body_);
    }

    virtual void visit(const Var &op) {}

    virtual void visit(const Store &op) {
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
        (*this)(op->expr_);
    }

    virtual void visit(const Alloc &op) {}

    virtual void visit(const Free &op) {}

    virtual void visit(const Load &op) {
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
    }

    virtual void visit(const ReduceTo &op) {
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
        (*this)(op->expr_);
    }

    virtual void visit(const IntConst &op) {}

    virtual void visit(const FloatConst &op) {}

    virtual void visit(const BoolConst &op) {}

    virtual void visit(const Add &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Sub &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Mul &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const RealDiv &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const FloorDiv &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const CeilDiv &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const RoundTowards0Div &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Mod &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Remainder &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Min &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Max &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const LT &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const LE &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const GT &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const GE &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const EQ &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const NE &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const LAnd &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const LOr &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const LNot &op) { (*this)(op->expr_); }

    virtual void visit(const Sqrt &op) { (*this)(op->expr_); }

    virtual void visit(const Exp &op) { (*this)(op->expr_); }

    virtual void visit(const Square &op) { (*this)(op->expr_); }

    virtual void visit(const Sigmoid &op) { (*this)(op->expr_); }

    virtual void visit(const Tanh &op) { (*this)(op->expr_); }

    virtual void visit(const Abs &op) { (*this)(op->expr_); }

    virtual void visit(const Floor &op) { (*this)(op->expr_); }

    virtual void visit(const Ceil &op) { (*this)(op->expr_); }

    virtual void visit(const For &op) {
        (*this)(op->begin_);
        (*this)(op->end_);
        (*this)(op->len_);
        for (auto &&r : op->property_->reductions_) {
            for (auto &&item : r->begins_) {
                (*this)(item);
            }
            for (auto &&item : r->ends_) {
                (*this)(item);
            }
        }
        (*this)(op->body_);
    }

    virtual void visit(const If &op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        if (op->elseCase_.isValid()) {
            (*this)(op->elseCase_);
        }
    }

    virtual void visit(const Assert &op) {
        (*this)(op->cond_);
        (*this)(op->body_);
    }

    virtual void visit(const Assume &op) {
        (*this)(op->cond_);
        (*this)(op->body_);
    }

    virtual void visit(const IfExpr &op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        (*this)(op->elseCase_);
    }

    virtual void visit(const Cast &op) { (*this)(op->expr_); }

    virtual void visit(const Intrinsic &op) {
        for (auto &&param : op->params_) {
            (*this)(param);
        }
    }

    virtual void visit(const Eval &op) { (*this)(op->expr_); }

    virtual void visit(const MatMul &op) {
        (*this)(op->a_);
        (*this)(op->b_);
        (*this)(op->c_);
        (*this)(op->alpha_);
        (*this)(op->beta_);
        (*this)(op->m_);
        (*this)(op->k_);
        (*this)(op->n_);
        (*this)(op->lda_);
        (*this)(op->ldb_);
        (*this)(op->ldc_);
        (*this)(op->stridea_);
        (*this)(op->strideb_);
        (*this)(op->stridec_);
        (*this)(op->batchSize_);
        (*this)(op->equivalent_);
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_VISITOR_H
