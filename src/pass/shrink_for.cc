#include <math/min_max.h>
#include <pass/shrink_for.h>
#include <pass/z3_simplify.h>

namespace ir {

Stmt ShrinkFor::visit(const For &_op) {
    auto var = makeVar(_op->iter_).as<VarNode>();
    auto hash = getHash(var);
    newRange_.erase(hash);

    iterStack_.emplace_back(var);
    defStack_.emplace_back(defs_);
    defs_.insert(_op->iter_);
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    defs_.erase(_op->iter_);
    defStack_.pop_back();
    iterStack_.pop_back();

    if (!newRange_.count(hash)) {
        return op->body_;
    }
    auto lower = makeMinMax(newRange_.at(hash).first);
    auto upper = makeMaxMin(newRange_.at(hash).second);

    if (op->property_.unroll_ ||
        (op->property_.parallel_.substr(0, 10) == "threadIdx." &&
         !op->property_.reductions_.empty())) {
        // Backends do not support these loops to be of variable lengths
        if (lower.isValid() && lower->nodeType() != ASTNodeType::IntConst) {
            return op;
        }
        if (upper.isValid() && upper->nodeType() != ASTNodeType::IntConst) {
            return op;
        }
    }

    if (op->step_->nodeType() == ASTNodeType::IntConst) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            if (lower.isValid()) {
                op->begin_ = lower;
            }
            if (upper.isValid()) {
                op->end_ = makeAdd(upper, op->step_);
            }
            op->len_ = makeFloorDiv(makeSub(op->end_, op->begin_), op->step_);
        } else if (step < 0) {
            if (upper.isValid()) {
                op->begin_ = upper;
            }
            if (lower.isValid()) {
                op->end_ = makeAdd(lower, op->step_);
            }
            op->len_ = makeFloorDiv(makeSub(op->end_, op->begin_), op->step_);
        }
    }

    return op;
}

Stmt ShrinkFor::visit(const VarDef &op) {
    defs_.insert(op->name_);
    auto ret = CompTransientBounds::visit(op);
    defs_.erase(op->name_);
    return ret;
}

Stmt shrinkFor(const Stmt &_op) {
    auto op = simplifyPass(_op); // Const prop + eliminate empty loops
    op = ShrinkFor()(op);
    return z3Simplify(op);
}

} // namespace ir

