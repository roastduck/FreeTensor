#ifndef MUTATOR_H_
#define MUTATOR_H_

#include <stdexcept>

#include "ASTNode.h"

class Mutator {
public:
#define DISPATCH_CASE(name) \
            case ASTNodeType::name: \
                return mutate(static_cast<name##Node*>(op.get())); \
                break;

    virtual std::shared_ptr<ExprNode> operator()(const std::shared_ptr<ExprNode> &op) {
        switch (op->nodeType()) {
            DISPATCH_CASE(Integer)
            DISPATCH_CASE(Var)
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
            DISPATCH_CASE(Assign)
            default:
                throw std::runtime_error("Unrecognized ASTNodeType");
        }
    }

    virtual std::shared_ptr<StmtNode> operator()(const std::shared_ptr<StmtNode> &op) {
        switch (op->nodeType()) {
            DISPATCH_CASE(StmtSeq)
            DISPATCH_CASE(VarDef)
            DISPATCH_CASE(Invoke)
            DISPATCH_CASE(IfThenElse)
            DISPATCH_CASE(While)
            DISPATCH_CASE(DoWhile)
            DISPATCH_CASE(For)
            DISPATCH_CASE(Return)
            DISPATCH_CASE(Break)
            DISPATCH_CASE(Continue)
            default:
                throw std::runtime_error("Unrecognized ASTNodeType");
        }
    }

    virtual std::shared_ptr<GlobalNode> operator()(const std::shared_ptr<GlobalNode> &op) {
        switch (op->nodeType()) {
            DISPATCH_CASE(Function)
            DISPATCH_CASE(GlobalVarDef)
            default:
                throw std::runtime_error("Unrecognized ASTNodeType");
        }
    }

#undef DISPATCH_CASE

    virtual std::shared_ptr<ProgramNode> operator()(const std::shared_ptr<ProgramNode> &op) {
        return mutate(op.get());
    }

protected:
    virtual std::shared_ptr<ProgramNode> mutate(const ProgramNode *op) {
        std::vector<std::shared_ptr<GlobalNode>> funcs;
        funcs.reserve(op->funcs_.size());
        for (auto &&func : op->funcs_) {
            funcs.push_back((*this)(func));
        }
        return ProgramNode::make(funcs);
    }

    virtual std::shared_ptr<GlobalNode> mutate(const FunctionNode *op) {
        curPath_ = op->name_ + "/";
        return FunctionNode::make(op->type_, op->name_, op->args_, op->body_ ? (*this)(op->body_) : nullptr);
    }

    virtual std::shared_ptr<GlobalNode> mutate(const GlobalVarDefNode *op) {
        return GlobalVarDefNode::make(op->type_, op->name_, op->init_);
    }

    virtual std::shared_ptr<StmtNode> mutate(const StmtSeqNode *op) {
        std::vector<std::shared_ptr<StmtNode>> stmts;
        stmts.reserve(op->stmts_.size());
        if (!op->isBlock_) {
            for (auto &&stmt : op->stmts_) {
                stmts.push_back((*this)(stmt));
            }
            return StmtSeqNode::make(stmts, op->isBlock_);
        }
        int oldLen = curPath_.length();
        curPath_ += std::to_string(blockCnt_++) + "/";
        for (auto &&stmt : op->stmts_) {
            stmts.push_back((*this)(stmt));
        }
        auto ret = StmtSeqNode::make(stmts, op->isBlock_);
        curPath_.resize(oldLen);
        return ret;
    }

    virtual std::shared_ptr<StmtNode> mutate(const VarDefNode *op) {
        return VarDefNode::make(op->type_, op->name_);
    }

    virtual std::shared_ptr<ExprNode> mutate(const AssignNode *op) {
        return AssignNode::make(op->var_, (*this)(op->expr_));
    }

    virtual std::shared_ptr<StmtNode> mutate(const IfThenElseNode *op) {
        auto cond = (*this)(op->cond_);
        auto thenCase = (*this)(op->thenCase_);
        auto elseCase = op->elseCase_ != nullptr ? (*this)(op->elseCase_) : nullptr;
        return IfThenElseNode::make(cond, thenCase, elseCase);
    }

    virtual std::shared_ptr<StmtNode> mutate(const WhileNode *op) {
        return WhileNode::make((*this)(op->cond_), (*this)(op->body_));
    }

    virtual std::shared_ptr<StmtNode> mutate(const DoWhileNode *op) {
        return DoWhileNode::make((*this)(op->cond_), (*this)(op->body_));
    }

    virtual std::shared_ptr<StmtNode> mutate(const ForNode *op) {
        return ForNode::make((*this)(op->init_), (*this)(op->cond_), (*this)(op->incr_), (*this)(op->body_));
    }

    virtual std::shared_ptr<StmtNode> mutate(const ReturnNode *op) {
        return ReturnNode::make((*this)(op->expr_));
    }

    virtual std::shared_ptr<StmtNode> mutate(const BreakNode *op) {
        return BreakNode::make();
    }

    virtual std::shared_ptr<StmtNode> mutate(const ContinueNode *op) {
        return ContinueNode::make();
    }

    virtual std::shared_ptr<StmtNode> mutate(const InvokeNode *op) {
        return InvokeNode::make((*this)(op->expr_));
    }

    virtual std::shared_ptr<ExprNode> mutate(const IntegerNode *op) {
        return IntegerNode::make(op->literal_);
    }

    virtual std::shared_ptr<ExprNode> mutate(const VarNode *op) {
        return VarNode::make(op->type_, op->name_);
    }

    virtual std::shared_ptr<ExprNode> mutate(const CallNode *op) {
        std::vector<std::shared_ptr<ExprNode>> args;
        args.reserve(op->args_.size());
        for (auto &&arg : op->args_) {
            args.push_back((*this)(arg));
        }
        return CallNode::make(op->type_, op->callee_, args);
    }

    virtual std::shared_ptr<ExprNode> mutate(const CastNode *op) {
        return CastNode::make(op->type_, (*this)(op->expr_));
    }

    virtual std::shared_ptr<ExprNode> mutate(const LNotNode *op) {
        return LNotNode::make((*this)(op->expr_));
    }

    virtual std::shared_ptr<ExprNode> mutate(const SelectNode *op) {
        auto cond = (*this)(op->cond_);
        auto thenCase = (*this)(op->thenCase_);
        auto elseCase = (*this)(op->elseCase_);
        return SelectNode::make(cond, thenCase, elseCase);
    }

#define VISIT_BINARY_NODE(name) \
    virtual std::shared_ptr<ExprNode> mutate(const name##Node *op) { \
        return name##Node::make((*this)(op->lhs_), (*this)(op->rhs_)); \
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

#endif  // MUTATOR_H_
