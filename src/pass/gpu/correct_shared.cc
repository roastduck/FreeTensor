#include <pass/gpu/correct_shared.h>

namespace ir {

namespace gpu {

Stmt CorrectShared::visit(const For &op) {
    if (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
        op->parallel_ == "threadIdx.z") {
        stack_.emplace_back(op);
        auto ret = Mutator::visit(op);
        stack_.pop_back();
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt CorrectShared::visit(const VarDef &_op) {
    if (_op->buffer_->mtype() == MemType::GPUShared) {
        int pos = defPos_[_op->name_] = stack_.size();
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        for (int i = pos - 1; i >= 0; i--) {
            op->buffer_ = op->buffer_.clone();
            auto &shape = op->buffer_->tensor().shape();
            shape.insert(shape.begin(), stack_[i]->infoLen_);
        }
        op->pinned_ = true;
        return op;
    } else {
        return Mutator::visit(_op);
    }
}

Expr CorrectShared::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return alterAccess(op);
}

Stmt CorrectShared::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return alterAccess(op);
}

Stmt CorrectShared::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return alterAccess(op);
}

} // namespace gpu

} // namespace ir

