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
};

} // namespace ir

#endif // VISITOR_H
