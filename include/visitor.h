#ifndef VISITOR_H
#define VISITOR_H

#include <functional>

#include <debug.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

class Visitor {
  public:
    virtual ~Visitor() {}

    virtual void operator()(const AST &op) final;

  protected:
    // Additional hook for any expressions
    virtual void visitExpr(const Expr &op,
                           const std::function<void(const Expr &)> &visitNode) {
        visitNode(op);
    }

    // Additional hook for any statements
    virtual void visitStmt(const Stmt &op,
                           const std::function<void(const Stmt &)> &visitNode) {
        visitNode(op);
    }

    // Hooks for each node
    virtual void visit(const Any &op) {}

    virtual void visit(const StmtSeq &op) {
        for (auto &&stmt : op->stmts_) {
            (*this)(stmt);
        }
    }

    virtual void visit(const VarDef &op) {
        for (auto &&dim : op->buffer_->tensor().shape()) {
            (*this)(dim);
        }
        (*this)(op->body_);
        if (op->infoAccLower_.isValid()) {
            for (auto &&index : *op->infoAccLower_) {
                (*this)(index);
            }
        }
        if (op->infoAccLen_.isValid()) {
            for (auto &&index : *op->infoAccLen_) {
                (*this)(index);
            }
        }
    }

    virtual void visit(const Var &op) {}

    virtual void visit(const Store &op) {
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
        (*this)(op->expr_);
    }

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

    virtual void visit(const Div &op) {
        (*this)(op->lhs_);
        (*this)(op->rhs_);
    }

    virtual void visit(const Mod &op) {
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

    virtual void visit(const For &op) {
        (*this)(op->begin_);
        (*this)(op->end_);
        (*this)(op->body_);
        if (op->infoLen_.isValid()) {
            (*this)(op->infoLen_);
        }
        if (op->infoMaxBegin_.isValid()) {
            (*this)(op->infoMaxBegin_);
        }
        if (op->infoMinEnd_.isValid()) {
            (*this)(op->infoMinEnd_);
        }
    }

    virtual void visit(const If &op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        if (op->elseCase_.isValid()) {
            (*this)(op->elseCase_);
        }
        if (op->infoNotCond_.isValid()) {
            (*this)(op->infoNotCond_);
        }
    }

    virtual void visit(const Assert &op) {
        (*this)(op->cond_);
        (*this)(op->body_);
    }

    virtual void visit(const Intrinsic &op) {
        for (auto &&param : op->params_) {
            (*this)(param);
        }
    }

    virtual void visit(const Eval &op) { (*this)(op->expr_); }
};

} // namespace ir

#endif // VISITOR_H
