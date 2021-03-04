#include <pass/shrink_var.h>
#include <pass/simplify.h>

namespace ir {

Stmt ShrinkVar::visit(const VarDef &_op) {
    if (_op->buffer_->atype() != AccessType::Cache) {
        return Mutator::visit(_op);
    }

    if (!newRange_.count(_op->id())) { // not used at all
        return (*this)(_op->body_);
    }
    auto &&range = newRange_.at(_op->id());

    offset_.erase(_op->name_);
    size_t n = _op->buffer_->tensor().shape().size();
    ASSERT(range.lower_.size() == n);
    ASSERT(range.len_.size() == n);
    offset_[_op->name_] = range.lower_;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    op->buffer_ = op->buffer_.clone();
    op->buffer_->tensor().setShape(range.len_);
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
    // (3) Modify var definitions
    // (4) Simplify the new indicies

    // (1)
    Stmt op;
    SimplifyPass::LowerBoundsMap lower;
    SimplifyPass::UpperBoundsMap upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds(_op);

    // (2)
    CompAccessBound visitor(lower, upper);
    visitor(op);

    // (3)
    op = ShrinkVar(visitor.results())(op);

    // (4)
    return simplifyPass(op);
}

} // namespace ir

