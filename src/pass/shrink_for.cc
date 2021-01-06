#include <analyze/comp_for_bound.h>
#include <pass/shrink_for.h>
#include <pass/simplify.h>

namespace ir {

Stmt ShrinkFor::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    op->begin_ = op->info_max_begin_;
    op->end_ = op->info_min_end_;
    op->info_max_begin_ = nullptr;
    op->info_min_end_ = nullptr;
    return op;
}

Stmt shrinkFor(const Stmt &_op) {
    // Algorithm:
    // (1) Simplify and get bounds of every iterators
    // (2) Represent the bounds of each iterators with min / max expressions
    // (3) Simplify those min / max expressions
    // (4) Modify For definitions
    // (5) Simplify the new indicies

    // (1)
    Stmt op;
    SimplifyPass::BoundsMap lower, upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds(_op);

    // (2)
    CompForBound marker(lower, upper);
    op = marker(op);

    // (3)
    op = simplifyPass(op);

    // (4)
    op = ShrinkFor()(op);

    // (5)
    return simplifyPass(op);
}

} // namespace ir

