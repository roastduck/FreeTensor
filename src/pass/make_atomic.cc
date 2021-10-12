#include <analyze/deps.h>
#include <pass/make_atomic.h>
#include <pass/make_reduction.h>

namespace ir {

void FindAllParallel::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    Visitor::visit(op);
    loopStack_.pop_back();

    if (!op->parallel_.empty()) {
        results_[op->id()] = {op->parallel_, loopStack_};
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

    std::vector<FindDepsCond> cond;
    FindAllParallel finder;
    finder(op);
    for (auto &&[loop, info] : finder.results()) {
        FindDepsCond findDepsCond{{loop, DepDirection::Different}};
        for (auto &&outerLoop : info.outerLoops_) {
            findDepsCond.push_back({outerLoop, DepDirection::Same});
        }
        cond.emplace_back(std::move(findDepsCond));
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
            finder.results().at(d.cond_[0].first.name_).type_.substr(0, 10) !=
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

