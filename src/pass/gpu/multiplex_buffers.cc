#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/gpu/multiplex_buffers.h>

namespace ir {

namespace gpu {

void FindParallelLoops::visit(const For &op) {
    if (op->property_.parallel_ == "threadIdx.x" ||
        op->property_.parallel_ == "threadIdx.y" ||
        op->property_.parallel_ == "threadIdx.z" ||
        op->property_.parallel_ == "blockIdx.x" ||
        op->property_.parallel_ == "blockIdx.y" ||
        op->property_.parallel_ == "blockIdx.z") {
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
            if (outer->property_.parallel_.substr(0, 10) == "threadIdx.") {
                affecting_[op->id()].insert(outer->id());
            }
        }
    } else if (op->buffer_->mtype() == MemType::GPUWarp) {
        for (auto &&outer : stack_) {
            if (outer->property_.parallel_.substr(0, 11) == "threadIdx.x") {
                //Only support conditions that threadIdx.x <= 32
                makeAssert(StmtNode::newId(), makeLE(outer->len_, makeIntConst(32)), makeStmtSeq(StmtNode::newId(), std::initializer_list<Stmt>()));
                affecting_[op->id()].insert(outer->id());
            }
        }
    }
}

Stmt MultiplexMutator::visit(const For &op) {
    if (op->property_.parallel_ == "threadIdx.x" ||
        op->property_.parallel_ == "threadIdx.y" ||
        op->property_.parallel_ == "threadIdx.z" ||
        op->property_.parallel_ == "blockIdx.x" ||
        op->property_.parallel_ == "blockIdx.y" ||
        op->property_.parallel_ == "blockIdx.z") {
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
                auto &shape = op->buffer_->tensor().shape();
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

Stmt multiplexBuffers(const Stmt &op) {
    FindParallelLoops finder;
    finder(op);

    // Criteria:
    // 1. There is dependencies
    // AND
    // 2. The value stored is loop-variant

    auto variantMap = findLoopVariance(op).second;

    std::vector<FindDepsCond> conds;
    std::unordered_map<ID, std::string> parallelScopes; // For ID -> parallel
    for (auto &&loop : finder.loops()) {
        conds.emplace_back(FindDepsCond{{loop->id(), DepDirection::Different}});
        parallelScopes[loop->id()] = loop->property_.parallel_;
    }
    std::unordered_map<ID, std::unordered_set<ID>>
        affecting; // VarDef ID -> For ID
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        return later.def_->buffer_->mtype() == MemType::GPUShared ||
               later.def_->buffer_->mtype() == MemType::GPUGlobal;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        if (finder.affecting().count(d.defId()) &&
            finder.affecting().at(d.defId()).count(d.cond_[0].first.id_)) {
            if (isVariant(variantMap, d.def(), d.cond_[0].first.id_)) {
                affecting[d.defId()].insert(d.cond_[0].first.id_);
            }
        }
    };
    findDeps(op, conds, found, FindDepsMode::Dep, DEP_ALL, filter, true, false);

    return MultiplexMutator(affecting)(op);
}

} // namespace gpu

} // namespace ir
