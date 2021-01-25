#include <analyze/comp_access_bound.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>

namespace ir {

Stmt ShrinkVar::visit(const VarDef &_op) {
    if (_op->buffer_->atype() != AccessType::Cache) {
        return Mutator::visit(_op);
    }

    offset_.erase(_op->name_);
    size_t n = _op->buffer_->tensor().shape().size();
    ASSERT(_op->info_acc_len_->size() == n);
    ASSERT(_op->info_acc_lower_->size() == n);
    offset_[_op->name_] = *_op->info_acc_lower_;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    op->buffer_ = op->buffer_.clone();
    auto &shape = op->buffer_->tensor().shape();
    for (size_t i = 0; i < n; i++) {
        shape[i] = (*op->info_acc_len_)[i];
    }

    op->info_acc_lower_ = nullptr;
    op->info_acc_len_ = nullptr;
    return op;
}

Expr ShrinkVar::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    return modifyAccess(__op.as<LoadNode>());
}

Stmt ShrinkVar::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    return modifyAccess(__op.as<StoreNode>());
}

Stmt ShrinkVar::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    return modifyAccess(__op.as<ReduceToNode>());
}

Stmt shrinkVar(const Stmt &_op) {
    // Algorithm:
    // (1) Simplify and get bounds of every expressions
    // (2) Represent the bounds of each vars with min / max expressions
    // (3) Simplify those min / max expressions
    // (4) Modify var definitions
    // (5) Simplify the new indicies

    // (1)
    Stmt op;
    SimplifyPass::BoundsMap lower, upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds(_op);

    // (2)
    CompAccessBound marker(lower, upper);
    op = marker(op);

    // (3)
    op = simplifyPass(op);

    // (4)
    op = ShrinkVar()(op);

    // (5)
    return simplifyPass(op);
}

} // namespace ir

