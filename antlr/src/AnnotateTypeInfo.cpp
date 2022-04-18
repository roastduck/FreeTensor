#include "AnnotateTypeInfo.h"
#include "error.h"

std::shared_ptr<ProgramNode> AnnotateTypeInfo::annotate(
        const std::shared_ptr<ProgramNode> &op, const std::unordered_map<std::string, ExprType> &types) {
    types_ = &types;
    return (*this)(op);
}

std::shared_ptr<GlobalNode> AnnotateTypeInfo::mutate(const FunctionNode *op) {
    curFunc_ = op->name_;
    return Mutator::mutate(op);
}

std::shared_ptr<ExprNode> AnnotateTypeInfo::mutate(const CallNode *op) {
    auto ret = AS(Mutator::mutate(op), Call);
    return CallNode::make(types_->at(ret->callee_), ret->callee_, ret->args_);
}

std::shared_ptr<ExprNode> AnnotateTypeInfo::mutate(const VarNode *op) {
    auto ret = AS(Mutator::mutate(op), Var);
    return VarNode::make(types_->at(getFullname(ret->name_)), ret->name_);
}

std::shared_ptr<ExprNode> AnnotateTypeInfo::mutate(const AssignNode *op) {
    auto ret = AS(Mutator::mutate(op), Assign);
    return AssignNode::make(ret->var_, CastNode::make(types_->at(getFullname(ret->var_)), ret->expr_));
}

std::shared_ptr<StmtNode> AnnotateTypeInfo::mutate(const ReturnNode *op) {
    auto ret = AS(Mutator::mutate(op), Return);
    return ReturnNode::make(CastNode::make(types_->at(curFunc_), ret->expr_));
}

std::string AnnotateTypeInfo::getFullname(const std::string &name) const {
    for (auto i = curPath_.length() - 1; ~i; i--) {
        if (curPath_[i] == '/') {
            auto fullname = curPath_.substr(0, i + 1) + name;
            if (types_->count(fullname)) {
                return fullname;
            }
        }
    }
    if (types_->count(name)) {
        return name;
    }
    throw std::runtime_error("name " + name + " not found");
}

