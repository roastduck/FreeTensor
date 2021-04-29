#include <analyze/find_loop_variance.h>
#include <pass/gpu/correct_shared.h>

namespace ir {

namespace gpu {

void FindAffectingLoops::visit(const For &op) {
    if (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
        op->parallel_ == "threadIdx.z") {
        loops_.insert(op->id());
        Visitor::visit(op);
        loops_.erase(op->id());
    } else {
        Visitor::visit(op);
    }
}

void FindAffectingLoops::visit(const VarDef &op) {
    if (op->buffer_->mtype() == MemType::GPUShared) {
        ASSERT(!defs_.count(op->name_));
        defs_[op->name_] = op->id();
        Visitor::visit(op);
        defs_.erase(op->name_);
    } else {
        Visitor::visit(op);
    }
}

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
        ASSERT(!defs_.count(_op->name_));
        defs_[_op->name_] = _op->id();
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        defs_.erase(_op->name_);

        if (affecting_.count(op->id())) {
            auto &&aff = affecting_.at(op->id());
            for (int i = pos - 1; i >= 0; i--) {
                if (aff.count(stack_[i]->id())) {
                    auto &shape = op->buffer_->tensor().shape();
                    shape.insert(shape.begin(), stack_[i]->len_);
                }
            }
            op->pinned_ = true;
        }
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

Stmt correctShared(const Stmt &op) {
    auto variants = findLoopVariance(op);
    FindAffectingLoops finder(variants);
    finder(op);
    return CorrectShared(finder.results())(op);
}

} // namespace gpu

} // namespace ir

