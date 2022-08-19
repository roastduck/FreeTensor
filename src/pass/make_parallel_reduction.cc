#include <itertools.hpp>

#include <analyze/analyze_linear.h>
#include <analyze/check_all_defined.h>
#include <analyze/deps.h>
#include <hash.h>
#include <math/min_max.h>
#include <pass/make_nested_loops.h>
#include <pass/make_parallel_reduction.h>
#include <pass/make_reduction.h>
#include <pass/simplify.h>

namespace freetensor {

static bool isDenseOver(const Expr &expr, const std::string &iter) {
    AnalyzeLinear analyzeLinear;
    analyzeLinear(expr);
    auto &&lin = analyzeLinear.result().at(expr);
    if (lin.coeff_.size() == 1 && std::abs(lin.coeff_.front().k_) == 1 &&
        lin.coeff_.front().a_->nodeType() == ASTNodeType::Var) {
        Var var = lin.coeff_.front().a_.as<VarNode>();
        return var->name_ == iter;
    }
    return false;
}

static MemType localMType(MemType mtype) {
    switch (mtype) {
    case MemType::CPU:
        return MemType::CPU;
    case MemType::GPULocal:
    case MemType::GPUShared:
    case MemType::GPUGlobal:
        return MemType::GPULocal;
    default:
        ASSERT(false);
    }
}

void FindAllParallel::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    Visitor::visit(op);
    loopStack_.pop_back();

    if (op->property_->parallel_ != serialScope) {
        results_[op->id()] = {op->property_->parallel_, loopStack_};
    }
}

void FindSerialLoopsOverReduce::visit(const For &op) {
    loopStack_.emplace_back(op);
    Visitor::visit(op);
    loopStack_.pop_back();
}

void FindSerialLoopsOverReduce::visit(const ReduceTo &op) {
    for (auto it = loopStack_.rbegin(); it != loopStack_.rend(); it++) {
        if ((*it)->property_->parallel_ != serialScope) {
            break;
        }
        results_[op->id()].emplace_back(*it);
    }
}

Stmt MakeParallelReduction::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (toAlter_.count(op->id())) {
        std::unordered_map<ID, std::vector<std::vector<Expr>>> lowerMap,
            upperMap; // loop ID -> [dim][bound]
        for (auto &&loopId : toAlter_.at(op->id())) {
            if (auto &&parallel = paraScopes_.at(loopId);
                std::holds_alternative<CUDAScope>(parallel) &&
                std::get<CUDAScope>(parallel).level_ == CUDAScope::Block) {
                // Race-free reduction among thread blocks are impossible
                goto use_atomic;
            }
            for (auto &&[i, idx, dim] :
                 iter::zip(iter::count(), _op->indices_,
                           buffer(_op->var_)->tensor()->shape())) {
                // use _op because isVariant needs it
                if (isVariant(variantMap_, idx, loopId)) {
                    goto use_atomic;
                }
                std::vector<Expr> dimLowers{makeIntConst(0)}, dimUppers{dim};
                for (auto &&item :
                     unique_.getDefinedLower(idx, scopeDefined_.at(loopId))) {
                    dimLowers.emplace_back(item.expr());
                }
                for (auto &&item :
                     unique_.getDefinedUpper(idx, scopeDefined_.at(loopId))) {
                    dimUppers.emplace_back(item.expr());
                }
                lowerMap[loopId].emplace_back(std::move(dimLowers));
                upperMap[loopId].emplace_back(std::move(dimUppers));
            }
        }
        for (auto &&loopId : toAlter_.at(op->id())) {
            const auto &lowers = lowerMap[loopId]; // [dim][bound]
            const auto &uppers = upperMap[loopId]; // [dim][bound]
            for (auto &[redOp, var, allLowers, allUppers] :
                 forReductions_[loopId]) {
                // allLowers, allUppers : [dim][access][bound]
                if (redOp == op->op_ && var == op->var_) {
                    ASSERT(allLowers.size() == lowers.size());
                    ASSERT(allUppers.size() == uppers.size());
                    for (auto &&[allLowersItem, lowersItem] :
                         iter::zip(allLowers, lowers)) {
                        allLowersItem.emplace_back(lowersItem);
                    }
                    for (auto &&[allUppersItem, uppersItem] :
                         iter::zip(allUppers, uppers)) {
                        allUppersItem.emplace_back(uppersItem);
                    }
                    goto done;
                }
            }
            {
                std::vector<std::vector<std::vector<Expr>>> allLowers(
                    lowers.size()),
                    allUppers(uppers.size());
                for (auto &&[allLowersItem, lowersItem] :
                     iter::zip(allLowers, lowers)) {
                    allLowersItem.emplace_back(lowersItem);
                }
                for (auto &&[allUppersItem, uppersItem] :
                     iter::zip(allUppers, uppers)) {
                    allUppersItem.emplace_back(uppersItem);
                }
                forReductions_[loopId].emplace_back(ReductionItemFactors{
                    op->op_, op->var_, std::move(allLowers),
                    std::move(allUppers)});
            }
        done:;
        }
        return op;

    use_atomic:
        // There will be no cross-thread dependencies except the reduction we
        // are working on (guranteed by schedule/parallelize). Therefore, We
        // can cache the variable being reduced, so it can be first reduced
        // serially inside a thread, before reduced to the finally target in an
        // atomic operation. We will cache over some serial inner loops, if
        // reduction is invariant to this loop, or if the loop densly iterates
        // over the reduction
        ID loopToCache;
        std::vector<bool> preserveDim(op->indices_.size(), false);
        if (serialOverRed_.count(op->id())) {
            for (auto &&loop : serialOverRed_.at(op->id())) {
                bool noPreserve = true;
                for (auto &&[idx, preserve] :
                     iter::zip(_op->indices_, preserveDim)) {
                    // use _op because isVariant needs it
                    if (isVariant(variantMap_, idx, loop->id())) {
                        if (isDenseOver(idx, loop->iter_)) {
                            preserve = true;
                            noPreserve = false;
                        } else {
                            goto found_loop_to_cache;
                        }
                    }
                }
                if (noPreserve) {
                    loopToCache = loop->id();
                }
            }
        }
    found_loop_to_cache:
        if (loopToCache.isValid()) {
            std::vector<Expr> newShape, newTargetIndices;
            op->var_ += ".atomic_cache." + toString(op->id());
            op->indices_ = {};
            for (auto &&[preserve, idx, dim] :
                 iter::zip(preserveDim, _op->indices_,
                           buffer(_op->var_)->tensor()->shape())) {
                if (preserve) {
                    op->indices_.emplace_back(idx);
                    newShape.emplace_back(dim);
                    newTargetIndices.emplace_back(nullptr);
                } else {
                    newTargetIndices.emplace_back(idx);
                }
            }
            cacheAtomic_[loopToCache].emplace_back(_op, newShape,
                                                   newTargetIndices);
        } else {
            op->atomic_ = true;
        }
    }
    return op;
}

Stmt MakeParallelReduction::visit(const For &_op) {
    ASSERT(!paraScopes_.count(_op->id()));
    paraScopes_[_op->id()] = _op->property_->parallel_;
    scopeDefined_[_op->id()] = names();
    auto __op = BaseClass::visit(_op);
    scopeDefined_.erase(_op->id());
    paraScopes_.erase(_op->id());

    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (forReductions_.count(op->id())) {
        for (auto &&[redOp, var, allLowers, allUppers] :
             forReductions_.at(op->id())) {
            std::vector<Expr> begins, ends;
            for (auto &&dimLowers : allLowers) {
                begins.emplace_back(makeMinMax(dimLowers));
            }
            for (auto &&dimUppers : allUppers) {
                ends.emplace_back(
                    makeAdd(makeMaxMin(dimUppers), makeIntConst(1)));
            }
            op->property_->reductions_.emplace_back(makeReductionItem(
                redOp, var, std::move(begins), std::move(ends)));
        }
    }

    if (cacheAtomic_.count(op->id())) {
        Stmt ret = op;
        for (auto &&[reduce, newShape, targetIndices] :
             cacheAtomic_.at(op->id())) {
            auto cacheName =
                reduce->var_ + ".atomic_cache." + toString(reduce->id());
            auto dtype = buffer(reduce->var_)->tensor()->dtype();
            auto mtype = localMType(buffer(reduce->var_)->mtype());
            std::vector<Expr> cacheIndices;
            for (size_t i = 0, j = 0, n = newShape.size(); i < n; i++) {
                cacheIndices.emplace_back(
                    makeVar(cacheName + ".i" + std::to_string(i)));
                while (targetIndices[j].isValid()) {
                    j++;
                    ASSERT(j < targetIndices.size());
                }
                targetIndices[j] = cacheIndices.back();
            }
            Stmt init = makeStore(cacheName, cacheIndices,
                                  neutralVal(dtype, reduce->op_));
            Stmt flush =
                makeReduceTo(reduce->var_, targetIndices, reduce->op_,
                             makeLoad(cacheName, cacheIndices, dtype), true);
            init = makeNestedLoops(
                cacheIndices, iter::repeat(makeIntConst(0)), newShape,
                iter::repeat(makeIntConst(1)), newShape,
                iter::repeat(Ref<ForProperty>::make()), init);
            flush = makeNestedLoops(
                cacheIndices, iter::repeat(makeIntConst(0)), newShape,
                iter::repeat(makeIntConst(1)), newShape,
                iter::repeat(Ref<ForProperty>::make()), flush);
            ret = makeVarDef(cacheName,
                             makeBuffer(makeTensor(newShape, dtype),
                                        AccessType::Cache, mtype),
                             nullptr, makeStmtSeq({init, ret, flush}), false);
        }
        return ret;
    } else {
        return op;
    }
}

Stmt makeParallelReduction(const Stmt &_op) {
    auto op = makeReduction(_op);

    std::vector<FindDepsDir> direction;
    FindAllParallel parallelFinder;
    parallelFinder(op);
    auto &&paraInfo = parallelFinder.results();
    for (auto &&[loop, info] : paraInfo) {
        FindDepsDir findDepsDir{{loop, DepDirection::Different}};
        for (auto &&outerLoop : info.outerLoops_) {
            findDepsDir.push_back({outerLoop, DepDirection::Same});
        }
        direction.emplace_back(std::move(findDepsDir));
    }

    std::unordered_map<ID, std::unordered_set<ID>> toAlter;
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() >= 1);
        ASSERT(d.dir_.front().first.isNode_);
        auto &&loopId = d.dir_.front().first.id_;
        if (auto &&parallel = paraInfo.at(loopId).type_;
            std::holds_alternative<CUDAScope>(parallel) &&
            std::get<CUDAScope>(parallel).level_ == CUDAScope::Thread &&
            d.later() != d.earlier()) {
            // No need to use atomic because we can sync
            return;
        }
        toAlter[d.later().as<ReduceToNode>()->id()].insert(loopId);
        toAlter[d.earlier().as<ReduceToNode>()->id()].insert(loopId);
    };
    FindDeps()
        .direction(direction)
        .filterLater([](const AccessPoint &later) {
            return later.op_->nodeType() == ASTNodeType::ReduceTo;
        })
        .filterEarlier([](const AccessPoint &earlier) {
            return earlier.op_->nodeType() == ASTNodeType::ReduceTo;
        })
        .ignoreReductionWAW(false)(op, found);

    FindSerialLoopsOverReduce serialFinder;
    serialFinder(op);

    auto [variantExprMap, variantVarMap] = findLoopVariance(op);

    op = MakeParallelReduction(toAlter, serialFinder.results(),
                               variantExprMap)(op);
    op = simplify(op);
    return op;
}

} // namespace freetensor
