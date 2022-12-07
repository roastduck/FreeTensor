#include <pass/rename_var.h>

namespace freetensor {

Stmt RenameVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (auto it = rename_.find(op->name_); it != rename_.end()) {
        op->name_ = it->second;
    }
    if (op->viewOf_.has_value()) {
        if (auto it = rename_.find(*op->viewOf_); it != rename_.end()) {
            op->viewOf_ = it->second;
        }
    }
    return op;
}

Expr RenameVar::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (auto it = rename_.find(op->var_); it != rename_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt RenameVar::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (auto it = rename_.find(op->var_); it != rename_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt RenameVar::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (auto it = rename_.find(op->var_); it != rename_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt RenameVar::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    for (auto &&red : op->property_->reductions_) {
        if (auto it = rename_.find(red->var_); it != rename_.end()) {
            red->var_ = it->second;
        }
    }
    return op;
}

Stmt RenameVar::visit(const Alloc &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Alloc);
    auto op = __op.as<AllocNode>();
    if (auto it = rename_.find(op->var_); it != rename_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt RenameVar::visit(const Free &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Free);
    auto op = __op.as<FreeNode>();
    if (auto it = rename_.find(op->var_); it != rename_.end()) {
        op->var_ = it->second;
    }
    return op;
}

} // namespace freetensor
