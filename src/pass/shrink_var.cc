#include <analyze/all_defs.h>
#include <pass/pb_simplify.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace ir {

Stmt ShrinkVar::visit(const VarDef &_op) {
    if (_op->buffer_->atype() != AccessType::Cache || _op->sizeLim_.isValid() ||
        _op->pinned_ || !newRange_.count(_op->id())) {
        return Mutator::visit(_op);
    }

    auto &&range = newRange_.at(_op->id());

    size_t n = _op->buffer_->tensor().shape().size();
    ASSERT(range.lower_.size() == n);
    ASSERT(range.len_.size() == n);
    ASSERT(!offset_.count(_op->name_));
    offset_[_op->name_] = range.lower_;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    op->buffer_ = op->buffer_.clone();
    op->buffer_->tensor().setShape(range.len_);
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
    // Algorithm:
    // (1) Simplify and get bounds of every expressions
    // (2) Represent the bounds of each vars with min / max expressions
    // (3) Modify var definitions
    // (4) Simplify the new indicies

    // (1)
    Stmt op;
    BuiltinSimplify::LowerBoundsMap lower;
    BuiltinSimplify::UpperBoundsMap upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds<PBSimplify>(_op);

    // (2)
    std::unordered_map<std::string, AccessBound> bounds;
    for (auto &&[varDefId, name] : allDefs(op, {AccessType::Cache})) {
        bounds[varDefId] = compAccessBound(op, varDefId, lower, upper);
    }

    // (3)
    op = ShrinkVar(bounds)(op);

    // (4)
    return z3Simplify(op); // Currently BuiltinSimplify is not sufficient
}

Stmt shrinkSingleVar(const Stmt &_op, const std::string &varDefId) {
    // (1)
    Stmt op;
    BuiltinSimplify::LowerBoundsMap lower;
    BuiltinSimplify::UpperBoundsMap upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds<PBSimplify>(_op);

    // (2)
    std::unordered_map<std::string, AccessBound> bounds;
    bounds[varDefId] = compAccessBound(op, varDefId, lower, upper);

    // (3)
    op = ShrinkVar(bounds)(op);

    // (4)
    return z3Simplify(op); // Currently BuiltinSimplify is not sufficient
}

} // namespace ir

