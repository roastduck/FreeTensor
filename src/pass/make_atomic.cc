#include <analyze/deps.h>
#include <pass/make_atomic.h>
#include <pass/make_reduction.h>

namespace ir {

void FindAllParallel::visit(const For &op) {
    Visitor::visit(op);
    if (!op->parallel_.empty()) {
        results_[op->id()] = op->parallel_;
    }
}

Stmt MakeAtomic::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (toAlter_.count(op->id())) {
        op->atomic_ = true;
    }
    return op;
}

Stmt makeAtomic(const Stmt &_op) {
    auto op = makeReduction(_op);

    std::vector<std::vector<std::pair<std::string, DepDirection>>> cond;
    FindAllParallel finder;
    finder(op);
    for (auto &&loop : finder.results()) {
        cond.push_back({{loop.first, DepDirection::Different}});
    }

    std::unordered_set<std::string> toAlter;
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        // TODO: Make these statements device agnostic
        // This condition is critical to compiling efficiency, so we put it here
        // rather than at `found` below
        if (later.buffer_->mtype() == MemType::GPULocal) {
            return false;
        }

        return later.op_->nodeType() == ASTNodeType::ReduceTo &&
               earlier.op_->nodeType() == ASTNodeType::ReduceTo;
    };
    auto found = [&](const Dependency &d) {
        // TODO: Make these statements device agnostic
        if (d.later_.buffer_->mtype() == MemType::GPUShared &&
            finder.results().at(d.cond_[0].first).substr(0, 10) !=
                "threadIdx.") {
            return;
        }

        toAlter.insert(d.later().as<ReduceToNode>()->id());
        toAlter.insert(d.earlier().as<ReduceToNode>()->id());
    };
    findDeps(op, cond, found, FindDepsMode::Dep, DEP_ALL, filter, false);

    return MakeAtomic(toAlter)(op);
}

} // namespace ir

