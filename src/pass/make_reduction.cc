#include <analyze/hash.h>
#include <pass/make_reduction.h>

namespace ir {

bool MakeReduction::isSameElem(const Store &s, const Load &l) {
    if (s->var_ != l->var_) {
        return false;
    }
    ASSERT(s->indices_.size() == l->indices_.size());
    for (size_t i = 0, iEnd = s->indices_.size(); i < iEnd; i++) {
        if (getHash(s->indices_[i]) != getHash(l->indices_[i])) {
            return false;
        }
    }
    return true;
}

Stmt MakeReduction::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::Add:
        return doMake<AddNode>(op, ReduceOp::Add);
    case ASTNodeType::Mul:
        return doMake<MulNode>(op, ReduceOp::Mul);
    case ASTNodeType::Min:
        return doMake<MinNode>(op, ReduceOp::Min);
    case ASTNodeType::Max:
        return doMake<MaxNode>(op, ReduceOp::Max);
    default:
        return op;
    }
}

} // namespace ir

