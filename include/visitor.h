#ifndef VISITOR_H
#define VISITOR_H

#include <except.h>
#include <expr.h>
#include <stmt.h>

namespace ir {

class Visitor {
  public:
    virtual ~Visitor() {}

    virtual void operator()(const AST &op) {
        switch (op->nodeType()) {

#define DISPATCH_CASE(name)                                                    \
    case ASTNodeType::name:                                                    \
        visit(op.as<name##Node>());                                            \
        break;

            DISPATCH_CASE(StmtSeq);
            DISPATCH_CASE(VarDef);
            DISPATCH_CASE(Var);
            DISPATCH_CASE(Store);
            DISPATCH_CASE(Load);
            DISPATCH_CASE(IntConst);
            DISPATCH_CASE(FloatConst);
            DISPATCH_CASE(Add);
            DISPATCH_CASE(Sub);
            DISPATCH_CASE(Mul);
            DISPATCH_CASE(Div);
            DISPATCH_CASE(Mod);
            DISPATCH_CASE(LT);
            DISPATCH_CASE(LE);
            DISPATCH_CASE(GT);
            DISPATCH_CASE(GE);
            DISPATCH_CASE(EQ);
            DISPATCH_CASE(NE);
            DISPATCH_CASE(For);
            DISPATCH_CASE(If);

        default:
            ERROR("Unexpected AST node type");
        }
    }

  protected:
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
    }

    virtual void visit(const Var &op) {}

    virtual void visit(const Store &op) {
        (*this)(op->var_);
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
        (*this)(op->expr_);
    }

    virtual void visit(const Load &op) {
        (*this)(op->var_);
        for (auto &&index : op->indices_) {
            (*this)(index);
        }
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

    virtual void visit(const For &op) {
        (*this)(op->begin_);
        (*this)(op->end_);
        (*this)(op->body_);
    }

    virtual void visit(const If &op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        if (op->elseCase_.isValid()) {
            (*this)(op->elseCase_);
        }
    }
};

} // namespace ir

#endif // VISITOR_H
