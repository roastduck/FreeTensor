#include <pass/gpu/correct_shared_and_global.h>

namespace ir {

namespace gpu {

void FindAffectingLoops::visit(const For &op) {
    bool sharedFlag =
        mode_ == MemType::GPUShared &&
        (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
         op->parallel_ == "threadIdx.z");
    bool globalFlag =
        mode_ == MemType::GPUGlobal &&
        (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
         op->parallel_ == "threadIdx.z" || op->parallel_ == "blockIdx.x" ||
         op->parallel_ == "blockIdx.y" || op->parallel_ == "blockIdx.z");
    if (sharedFlag || globalFlag) {
        loops_.insert(op->id());
        Visitor::visit(op);
        loops_.erase(op->id());
    } else {
        Visitor::visit(op);
    }
}

void FindAffectingLoops::visit(const VarDef &op) {
    if (op->buffer_->mtype() == mode_) {
        ASSERT(!defs_.count(op->name_));
        defs_[op->name_] = op->id();
        Visitor::visit(op);
        defs_.erase(op->name_);
    } else {
        Visitor::visit(op);
    }
}

Stmt CorrectMutator::visit(const For &op) {
    bool sharedFlag =
        mode_ == MemType::GPUShared &&
        (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
         op->parallel_ == "threadIdx.z");
    bool globalFlag =
        mode_ == MemType::GPUGlobal &&
        (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
         op->parallel_ == "threadIdx.z" || op->parallel_ == "blockIdx.x" ||
         op->parallel_ == "blockIdx.y" || op->parallel_ == "blockIdx.z");
    if (sharedFlag || globalFlag) {
        stack_.emplace_back(op);
        auto ret = Mutator::visit(op);
        stack_.pop_back();
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt CorrectMutator::visit(const VarDef &_op) {
    if (_op->buffer_->mtype() == mode_) {
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

Expr CorrectMutator::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return alterAccess(op);
}

Stmt CorrectMutator::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return alterAccess(op);
}

Stmt CorrectMutator::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return alterAccess(op);
}

Stmt correctSharedAndGlobal(const Stmt &_op) {
    auto op = _op;
    auto variants = findLoopVariance(op);
    FindAffectingLoops sharedFinder(variants.first, MemType::GPUShared);
    sharedFinder(op);
    op = CorrectMutator(sharedFinder.results(), MemType::GPUShared)(op);
    FindAffectingLoops globalFinder(variants.first, MemType::GPUGlobal);
    globalFinder(op);
    op = CorrectMutator(globalFinder.results(), MemType::GPUGlobal)(op);
    return op;
}

} // namespace gpu

} // namespace ir

