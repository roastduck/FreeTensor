#include <stdexcept>

#include "VarAllocVisitor.h"

VarAllocVisitor::Map<int> VarAllocVisitor::allocVar(const std::shared_ptr<ASTNode> &op) {
    (*this)(op);
    return varMap_;
}

void VarAllocVisitor::visit(const FunctionNode *op) {
    offset_ = 0;
    for (auto &&arg : op->args_) {
        varMap_[op->name_ + "/" + arg.second] = offset_++;  // copy the args to another var
    }
    Visitor::visit(op);
}

void VarAllocVisitor::visit(const VarDefNode *op) {
    if (!varMap_.count(curPath_ + op->name_)) {
        varMap_[curPath_ + op->name_] = offset_++;
    } else {
        throw std::runtime_error("Var " + op->name_ + " is already defined");
    }
    Visitor::visit(op);
}

