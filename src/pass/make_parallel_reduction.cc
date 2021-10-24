#include <analyze/check_all_defined.h>
#include <analyze/deps.h>
#include <pass/make_parallel_reduction.h>
#include <pass/make_reduction.h>

namespace ir {

void FindAllParallel::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    Visitor::visit(op);
    loopStack_.pop_back();

    if (!op->property_.parallel_.empty()) {
        results_[op->id()] = {op->property_.parallel_, loopStack_};
    }
}

Stmt MakeParallelReduction::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (toAlter_.count(op->id())) {
        std::vector<SubTree<ExprNode, Nullable>> indices;
        for (auto &&loopId : toAlter_.at(op->id())) {
            if (paraScopes_.at(loopId).substr(0, 9) == "blockIdx.") {
                // Race-free reduction among thread blocks are impossible
                goto use_atomic;
            }
            for (auto &&idx : _op->indices_) {
                // use _op because isVariant needs it
                if (isVariant(variantMap_, idx, loopId)) {
                    goto use_atomic;
                }
                if (!checkAllDefined(scopeDefined_.at(loopId), idx)) {
                    indices.emplace_back(nullptr);
                } else {
                    indices.emplace_back(idx);
                }
            }
        }
        for (auto &&loopId : toAlter_.at(op->id())) {
            // TODO: Check for duplication
            forReductions_[loopId].emplace_back(
                ReductionItem{op->op_, op->var_, std::move(indices)});
        }
        return op;

    use_atomic:
        op->atomic_ = true;
    }
    return op;
}

Stmt MakeParallelReduction::visit(const For &_op) {
    ASSERT(!defined_.count(_op->iter_));
    ASSERT(!paraScopes_.count(_op->id()));
    defined_.insert(_op->iter_);
    paraScopes_[_op->id()] = _op->property_.parallel_;
    scopeDefined_[_op->id()] = defined_;
    auto __op = Mutator::visit(_op);
    scopeDefined_.erase(_op->id());
    paraScopes_.erase(_op->id());
    defined_.erase(_op->iter_);

    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (forReductions_.count(op->id())) {
        for (auto &&reduction : forReductions_.at(op->id())) {
            op->property_.reductions_.emplace_back(reduction);
        }
    }
    return op;
}

Stmt MakeParallelReduction::visit(const VarDef &op) {
    ASSERT(!defined_.count(op->name_));
    defined_.insert(op->name_);
    auto ret = Mutator::visit(op);
    defined_.erase(op->name_);
    return ret;
}

Stmt makeParallelReduction(const Stmt &_op) {
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

    std::unordered_map<std::string, std::unordered_set<std::string>> toAlter;
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        return later.op_ == earlier.op_ &&
               later.op_->nodeType() == ASTNodeType::ReduceTo;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() >= 1);
        ASSERT(d.cond_.front().first.isNode_);
        auto &&loopId = d.cond_.front().first.name_;
        toAlter[d.later().as<ReduceToNode>()->id()].insert(loopId);
    };
    findDeps(op, cond, found, FindDepsMode::Dep, DEP_ALL, filter, false);

    auto [variantExprMap, variantVarMap] = findLoopVariance(op);

    return MakeParallelReduction(toAlter, variantExprMap)(op);
}

} // namespace ir

