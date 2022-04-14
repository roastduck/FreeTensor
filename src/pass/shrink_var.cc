#include <analyze/all_defs.h>
#include <pass/shrink_var.h>
#include <pass/z3_simplify.h>

namespace ir {

Stmt ShrinkVar::visit(const VarDef &_op) {
    if (_op->buffer_->atype() != AccessType::Cache || _op->sizeLim_.isValid() ||
        _op->pinned_ || !newRange_.count(_op->id())) {
        return Mutator::visit(_op);
    }

    auto &&range = newRange_.at(_op->id());

    size_t n = _op->buffer_->tensor()->shape().size();
    ASSERT(range.lower_.size() == n);
    ASSERT(range.len_.size() == n);
    ASSERT(!offset_.count(_op->name_));
    offset_[_op->name_] = range.lower_;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    op->buffer_->tensor()->setShape(range.len_);
    offset_.erase(_op->name_);
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
    auto op = _op;

    // Algorithm:
    // (1) Represent the bounds of each vars with min / max expressions
    // (2) Modify var definitions
    // (3) Simplify the new indicies

    // (1)
    std::unordered_map<ID, AccessBound> bounds;
    for (auto &&[varDefId, name] : allDefs(op, {AccessType::Cache})) {
        bounds[varDefId] = compAccessBound(op, varDefId);
    }

    // (2)
    op = ShrinkVar(bounds)(op);

    // (3)
    return z3Simplify(op); // Currently BuiltinSimplify is not sufficient
}

Stmt shrinkSingleVar(const Stmt &_op, const ID &varDefId) {
    auto op = _op;

    // (1)
    std::unordered_map<ID, AccessBound> bounds;
    bounds[varDefId] = compAccessBound(op, varDefId);

    // (2)
    op = ShrinkVar(bounds)(op);

    // (3)
    return z3Simplify(op); // Currently BuiltinSimplify is not sufficient
}

} // namespace ir
