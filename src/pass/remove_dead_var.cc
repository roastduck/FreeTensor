#include <pass/remove_dead_var.h>

namespace freetensor {

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
    if (destination_ != op->var_) {
        uses_.insert(op->var_);
    }
    return op;
}

Stmt RemoveDeadVar::visit(const Store &op) {
    destination_ = op->var_;
    auto ret = Mutator::visit(op);
    destination_.clear();
    return ret;
}

Stmt RemoveDeadVar::visit(const ReduceTo &op) {
    destination_ = op->var_;
    auto ret = Mutator::visit(op);
    destination_.clear();
    return ret;
}

Stmt RemoveDeadVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->atype() == AccessType::Cache && !uses_.count(op->name_)) {
        isFixPoint_ = false;
        return RemoveAllWrites(op->name_)(op->body_);
    }

    uses_.erase(_op->name_);
    return op;
}

Stmt removeDeadVar(const Stmt &_op) {
    auto op = _op;
    for (int i = 0;; i++) {
        RemoveDeadVar mutator;
        op = mutator(op);
        if (mutator.isFixPoint() || i > 100) {
            if (i > 100) {
                WARNING("removeDeadVar iterates over 100 rounds. Maybe there "
                        "is a bug");
            }
            break;
        }
    }
    return op;
}

} // namespace freetensor
