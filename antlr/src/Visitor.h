#ifndef VISITOR_H_
#define VISITOR_H_

#include <stdexcept>

#include "ASTNode.h"

class Visitor {
public:
    virtual void operator()(const std::shared_ptr<ASTNode> &op) {
        switch (op->nodeType()) {

#define DISPATCH_CASE(name) \
            case ASTNodeType::name: \
                visit(static_cast<name##Node*>(op.get())); \
                break;

            DISPATCH_CASE(Program)
            DISPATCH_CASE(Function)
            DISPATCH_CASE(GlobalVarDef)
            DISPATCH_CASE(StmtSeq)
            DISPATCH_CASE(Integer)
            DISPATCH_CASE(Var)
            DISPATCH_CASE(VarDef)
            DISPATCH_CASE(Assign)
            DISPATCH_CASE(Invoke)
            DISPATCH_CASE(IfThenElse)
            DISPATCH_CASE(While)
            DISPATCH_CASE(DoWhile)
            DISPATCH_CASE(For)
            DISPATCH_CASE(Return)
            DISPATCH_CASE(Break)
            DISPATCH_CASE(Continue)
            DISPATCH_CASE(Call)
            DISPATCH_CASE(Cast)
            DISPATCH_CASE(Add)
            DISPATCH_CASE(Sub)
            DISPATCH_CASE(Mul)
            DISPATCH_CASE(Div)
            DISPATCH_CASE(Mod)
            DISPATCH_CASE(BAnd)
            DISPATCH_CASE(BOr)
            DISPATCH_CASE(BXor)
            DISPATCH_CASE(SLL)
            DISPATCH_CASE(SRA)
            DISPATCH_CASE(LNot)
            DISPATCH_CASE(LT)
            DISPATCH_CASE(LE)
            DISPATCH_CASE(GT)
            DISPATCH_CASE(GE)
            DISPATCH_CASE(EQ)
            DISPATCH_CASE(NE)
            DISPATCH_CASE(LAnd)
            DISPATCH_CASE(LOr)
            DISPATCH_CASE(Select)

#undef DISPATCH_CASE

            default:
                throw std::runtime_error("Unrecognized ASTNodeType");
        }
    }

protected:
    virtual void visit(const ProgramNode *op) {
        for (auto &&func : op->funcs_) {
            (*this)(func);
        }
    }

    virtual void visit(const FunctionNode *op) {
        curPath_ = op->name_ + "/";
        if (op->body_) {
            (*this)(op->body_);
        }
    }

    virtual void visit(const GlobalVarDefNode *op) {}

    virtual void visit(const StmtSeqNode *op) {
        if (!op->isBlock_) {
            for (auto &&stmt : op->stmts_) {
                (*this)(stmt);
            }
            return;
        }
        int oldLen = curPath_.length();
        curPath_ += std::to_string(blockCnt_++) + "/";
        for (auto &&stmt : op->stmts_) {
            (*this)(stmt);
        }
        curPath_.resize(oldLen);
    }

    virtual void visit(const IntegerNode *op) {}

    virtual void visit(const VarNode *op) {}

    virtual void visit(const VarDefNode *op) {}

    virtual void visit(const AssignNode *op) {
        (*this)(op->expr_);
    }

    virtual void visit(const IfThenElseNode *op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        if (op->elseCase_ != nullptr) {
            (*this)(op->elseCase_);
        }
    }

    virtual void visit(const WhileNode *op) {
        (*this)(op->cond_);
        (*this)(op->body_);
    }

    virtual void visit(const DoWhileNode *op) {
        (*this)(op->cond_);
        (*this)(op->body_);
    }

    virtual void visit(const ForNode *op) {
        (*this)(op->init_);
        (*this)(op->cond_);
        (*this)(op->incr_);
        (*this)(op->body_);
    }

    virtual void visit(const ReturnNode *op) {
        (*this)(op->expr_);
    }

    virtual void visit(const BreakNode *op) {}

    virtual void visit(const ContinueNode *op) {}

    virtual void visit(const InvokeNode *op) {
        (*this)(op->expr_);
    }

    virtual void visit(const CallNode *op) {
        for (auto &&arg : op->args_) {
            (*this)(arg);
        }
    }

    virtual void visit(const CastNode *op) {
        (*this)(op->expr_);
    }

    virtual void visit(const LNotNode *op) {
        (*this)(op->expr_);
    }

    virtual void visit(const SelectNode *op) {
        (*this)(op->cond_);
        (*this)(op->thenCase_);
        (*this)(op->elseCase_);
    }

#define VISIT_BINARY_NODE(name) \
    virtual void visit(const name##Node *op) { \
        (*this)(op->lhs_); \
        (*this)(op->rhs_); \
    }
    VISIT_BINARY_NODE(Add)
    VISIT_BINARY_NODE(Sub)
    VISIT_BINARY_NODE(Mul)
    VISIT_BINARY_NODE(Div)
    VISIT_BINARY_NODE(Mod)
    VISIT_BINARY_NODE(BAnd)
    VISIT_BINARY_NODE(BOr)
    VISIT_BINARY_NODE(BXor)
    VISIT_BINARY_NODE(SLL)
    VISIT_BINARY_NODE(SRA)
    VISIT_BINARY_NODE(LT)
    VISIT_BINARY_NODE(LE)
    VISIT_BINARY_NODE(GT)
    VISIT_BINARY_NODE(GE)
    VISIT_BINARY_NODE(EQ)
    VISIT_BINARY_NODE(NE)
    VISIT_BINARY_NODE(LAnd)
    VISIT_BINARY_NODE(LOr)
#undef VISIT_BINARY_NODE

    std::string curPath_;
    int blockCnt_;
};

#endif  // VISITOR_H_
