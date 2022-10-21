#ifdef FT_WITH_CUDA

#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/gpu/multiplex_buffers.h>

namespace freetensor {

namespace gpu {

void FindParallelLoops::visit(const For &op) {
    if (std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
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
            if (std::holds_alternative<CUDAScope>(
                    outer->property_->parallel_) &&
                std::get<CUDAScope>(outer->property_->parallel_).level_ ==
                    CUDAScope::Thread) {
                affecting_[op->id()].insert(outer->id());
            }
        }
    } else if (op->buffer_->mtype() == MemType::GPUWarp) {
        for (auto &&outer : stack_) {
            if (outer->property_->parallel_ == threadIdxX) {
                // Only support conditions that threadIdx.x <= warpSize
                makeAssert(
                    makeLE(outer->len_, makeIntConst(target_->warpSize())),
                    makeStmtSeq(std::initializer_list<Stmt>()));
                affecting_[op->id()].insert(outer->id());
            }
        }
    }
}

Stmt MultiplexMutator::visit(const For &op) {
    if (std::holds_alternative<CUDAScope>(op->property_->parallel_) ||
        std::holds_alternative<CUDAStreamScope>(op->property_->parallel_)) {
        stack_.emplace_back(op);
        auto ret = BaseClass::visit(op);
        stack_.pop_back();
        return ret;
    } else {
        return BaseClass::visit(op);
    }
}

Stmt MultiplexMutator::visit(const VarDef &_op) {
    int pos = defPos_[_op->name_] = stack_.size();
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (affecting_.count(op->id())) {
        auto &&aff = affecting_.at(op->id());
        for (int i = pos - 1; i >= 0; i--) {
            if (aff.count(stack_[i]->id())) {
                auto &shape = op->buffer_->tensor()->shape();
                shape.insert(shape.begin(), stack_[i]->len_);
            }
        }
        op->pinned_ = true;
    }
    return op;
}

Expr MultiplexMutator::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return alterAccess(op);
}

Stmt MultiplexMutator::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return alterAccess(op);
}

Stmt MultiplexMutator::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return alterAccess(op);
}

Stmt multiplexBuffers(const Stmt &op, const Ref<GPUTarget> &target) {
    FindParallelLoops finder(target);
    finder(op);

    std::unordered_map<ID, std::unordered_set<ID>> affecting,
        newAffecting; // VarDef ID -> For ID
    affecting = finder.affecting();

    // Criteria:
    // 1. The value stored is loop-variant
    // AND
    // 2. There is dependence

    // 1. The value stored is loop-variant
    auto variantMap = findLoopVariance(op).second;
    for (auto &&[vardef, loops] : affecting) {
        for (auto &&loop : loops) {
            if (isVariant(variantMap, vardef, loop)) {
                newAffecting[vardef].insert(loop);
            }
        }
    }
    affecting = std::move(newAffecting);

    // 2. There is dependence
    std::vector<FindDepsDir> direction;
    for (auto &&loop : finder.loops()) {
        direction.emplace_back(
            FindDepsDir{{loop->id(), DepDirection::Different}});
    }
    FindDeps()
        .direction(direction)
        .filterAccess([&](const AccessPoint &acc) {
            return affecting.count(acc.def_->id());
        })
        .eraseOutsideVarDef(false)(op, [&](const Dependency &d) {
            ASSERT(d.dir_.size() == 1);
            if (affecting.count(d.defId()) &&
                affecting.at(d.defId()).count(d.dir_[0].first.id_)) {
                newAffecting[d.defId()].insert(d.dir_[0].first.id_);
            }
        });
    affecting = std::move(newAffecting);

    return MultiplexMutator(affecting)(op);
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
