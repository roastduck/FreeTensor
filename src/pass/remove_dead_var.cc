#include <pass/remove_dead_var.h>

namespace ir {

Stmt RemoveAllWrites::visit(const Store &op) {
    return var_ == op->var_ ? makeStmtSeq("", {}) : Mutator::visit(op);
}

Stmt RemoveAllWrites::visit(const ReduceTo &op) {
    return var_ == op->var_ ? makeStmtSeq("", {}) : Mutator::visit(op);
}

Expr RemoveDeadVar::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    uses_.insert(op->var_);
    return op;
}

Stmt RemoveDeadVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->atype() == AccessType::Cache && !uses_.count(op->name_)) {
        return RemoveAllWrites(op->name_)(op->body_);
    }

    uses_.erase(_op->name_);
    return op;
}

} // namespace ir

