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

    ASSERT(newRange_.count(hash));
    auto newBegin = makeMinMax(newRange_.at(hash).first);
    auto newEndMinus1 = makeMaxMin(newRange_.at(hash).second);
    if (newBegin.isValid()) {
        op->begin_ = newBegin;
    }
    if (newEndMinus1.isValid()) {
        op->end_ = makeAdd(newEndMinus1, makeIntConst(1));
    }
    op->len_ = makeSub(op->end_, op->begin_);

    return op;
}

Stmt ShrinkFor::visit(const VarDef &op) {
    defs_.insert(op->name_);
    auto ret = CompTransientBounds::visit(op);
    defs_.erase(op->name_);
    return ret;
}

Stmt shrinkFor(const Stmt &_op, bool keepConst) {
    auto op = simplifyPass(_op); // Const prop + eliminate empty loops
    op = ShrinkFor(keepConst)(op);
    return z3Simplify(op);
}

} // namespace ir

