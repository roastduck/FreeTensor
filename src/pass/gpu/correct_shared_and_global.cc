#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/gpu/correct_shared_and_global.h>

namespace ir {

namespace gpu {

void FindParallelLoops::visit(const For &op) {
    if (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
        op->parallel_ == "threadIdx.z" || op->parallel_ == "blockIdx.x" ||
        op->parallel_ == "blockIdx.y" || op->parallel_ == "blockIdx.z") {
        loops_.emplace_back(op);
        stack_.emplace_back(op);
        Visitor::visit(op);
        stack_.pop_back();
    } else {
        Visitor::visit(op);
    }
}

void FindParallelLoops::visit(const VarDef &op) {
    Visitor::visit(op);
    if (op->buffer_->mtype() == MemType::GPUGlobal) {
        for (auto &&outer : stack_) {
            affecting_[op->id()].insert(outer->id());
        }
    } else if (op->buffer_->mtype() == MemType::GPUShared) {
        for (auto &&outer : stack_) {
            if (outer->parallel_.substr(0, 10) == "threadIdx.") {
                affecting_[op->id()].insert(outer->id());
            }
        }
    }
}

Stmt CorrectMutator::visit(const For &op) {
    if (op->parallel_ == "threadIdx.x" || op->parallel_ == "threadIdx.y" ||
        op->parallel_ == "threadIdx.z" || op->parallel_ == "blockIdx.x" ||
        op->parallel_ == "blockIdx.y" || op->parallel_ == "blockIdx.z") {
        stack_.emplace_back(op);
        auto ret = Mutator::visit(op);
        stack_.pop_back();
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt CorrectMutator::visit(const VarDef &_op) {
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

Stmt correctSharedAndGlobal(const Stmt &op) {
    FindParallelLoops finder;
    finder(op);

    // Criteria:
    // 1. There is dependencies
    // AND
    // 2. The value stored is loop-variant

    auto variantMap = findLoopVariance(op).second;

    std::vector<FindDepsCond> conds;
    std::unordered_map<std::string, std::string>
        parallelScopes; // For ID -> parallel
    for (auto &&loop : finder.loops()) {
        conds.emplace_back(FindDepsCond{{loop->id(), DepDirection::Different}});
        parallelScopes[loop->id()] = loop->parallel_;
    }
    std::unordered_map<std::string, std::unordered_set<std::string>>
        affecting; // VarDef ID -> For ID
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        return later.def_->buffer_->mtype() == MemType::GPUShared ||
               later.def_->buffer_->mtype() == MemType::GPUGlobal;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        if (finder.affecting().count(d.defId()) &&
            finder.affecting().at(d.defId()).count(d.cond_[0].first)) {
            if (isVariant(variantMap, d.def(), d.cond_[0].first)) {
                affecting[d.defId()].insert(d.cond_[0].first);
            }
        }
    };
    findDeps(op, conds, found, FindDepsMode::Dep, DEP_ALL, filter);

    return CorrectMutator(affecting)(op);
}

} // namespace gpu

} // namespace ir

