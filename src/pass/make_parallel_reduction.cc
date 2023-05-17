#include <analyze/analyze_linear.h>
#include <analyze/check_all_defined.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/deps.h>
#include <container_utils.h>
#include <hash.h>
#include <math/min_max.h>
#include <math/utils.h>
#include <pass/const_fold.h>
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

Stmt MakeLoopCarriedReduction::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (toAlter_.count(op->id())) {
        // 1. Check whether we have to use synced reduction
        for (auto &&loopId : toAlter_.at(op->id())) {
            for (auto &&[i, idx] : views::zip(
                     views::ints(0, ranges::unreachable), _op->indices_)) {
                if (isVariant(variantMap_, {idx, op}, loopId)) {
                    // Use sync
                    toUseSync_[op->id()] = UseSyncInfo{true};
                    return op;
                }
            }
        }
        for (auto &&loopId : toAlter_.at(op->id())) {
            // After the random access check, to make `toUseSync_` correct
            if (auto &&parallel = paraScopes_.at(loopId);
                std::holds_alternative<CUDAScope>(parallel) &&
                std::get<CUDAScope>(parallel).level_ == CUDAScope::Block) {
                // Race-free reduction among thread blocks are impossible. Use
                // sync
                toUseSync_[op->id()] = UseSyncInfo{false};
                return op;
            }
        }

        // 2. Compute range of loop-carried reduction
        CompUniqueBounds unique(*this);
        std::unordered_map<ID, std::vector<std::vector<Expr>>> lowerMap,
            upperMap; // loop ID -> [dim][bound]
        for (auto &&loopId : toAlter_.at(op->id())) {
            for (auto &&[i, idx, dim] :
                 views::zip(views::ints(0, ranges::unreachable), _op->indices_,
                            buffer(_op->var_)->tensor()->shape())) {
                std::vector<Expr> dimLowers{makeIntConst(0)}, dimUppers{dim};
                for (auto &&item :
                     unique.getDefinedLower(idx, scopeDefined_.at(loopId))) {
                    dimLowers.emplace_back(item.expr());
                }
                for (auto &&item :
                     unique.getDefinedUpper(idx, scopeDefined_.at(loopId))) {
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
                         views::zip(allLowers, lowers)) {
                        allLowersItem.emplace_back(lowersItem);
                    }
                    for (auto &&[allUppersItem, uppersItem] :
                         views::zip(allUppers, uppers)) {
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
                     views::zip(allLowers, lowers)) {
                    allLowersItem.emplace_back(lowersItem);
                }
                for (auto &&[allUppersItem, uppersItem] :
                     views::zip(allUppers, uppers)) {
                    allUppersItem.emplace_back(uppersItem);
                }
                forReductions_[loopId].emplace_back(ReductionItemFactors{
                    op->op_, op->var_, std::move(allLowers),
                    std::move(allUppers)});
            }
        done:;
        }
        return op;
    }
    return op;
}

Stmt MakeLoopCarriedReduction::visit(const For &_op) {
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

    return op;
}

bool MakeSyncReduction::canResideInGPULocal(DataType dtype,
                                            const std::vector<Expr> &shape,
                                            bool isRandomAccess) const {
#ifdef FT_WITH_CUDA
    if (isRandomAccess) {
        // Case 3: Random access
        return false;
    }

    // GPU registers are 32-bit
    int64_t sizeIn32Bit = 1;
    int64_t sizeInBytes = 1;
    for (auto &&dim : shape) {
        if (dim->nodeType() == ASTNodeType::IntConst) {
            sizeIn32Bit *=
                dim.as<IntConstNode>()->val_ * ceilDiv(sizeOf(dtype), 4);
            sizeInBytes *= dim.as<IntConstNode>()->val_ * sizeOf(dtype);
        } else {
            // Case 2: Dynamic shape
            return false;
        }
    }
    if (target_ == nullptr) {
        // Debug lowering without target info
        return true;
    } else {
        ASSERT(target_->type() == TargetType::GPU);
        auto gpu = target_.as<GPUTarget>();

        // Case 1a: Shape larger than register count
        if (sizeIn32Bit * gpuThreadDim_ > gpu->regsPerBlock()) {
            return false;
        }

        // Case 1b: Shape cannot be held in local memory
        if (sizeInBytes > gpu->maxLocalMemorySizePerThread()) {
            return false;
        }

        return true;
    }
#else
    return false;
#endif
}

MemType MakeSyncReduction::localMType(MemType mtype, DataType dtype,
                                      const std::vector<Expr> &shape,
                                      bool isRandomAccess) const {
    switch (mtype) {
    case MemType::CPU:
        return MemType::CPU;
    case MemType::GPULocal:
    case MemType::GPUShared:
    case MemType::GPUGlobal:
        return canResideInGPULocal(dtype, shape, isRandomAccess)
                   ? MemType::GPULocal
                   : MemType::GPUGlobal;
    default:
        ASSERT(false);
    }
}

Stmt MakeSyncReduction::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (auto it = toUseSync_.find(op->id()); it != toUseSync_.end()) {
        auto isRandomAccess = it->second.isRandomAccess_;

        // There will be no cross-thread dependences except the reduction we
        // are working on (guranteed by schedule/parallelize). Therefore, We
        // can cache the variable being reduced, so it can be first reduced
        // serially inside a thread, before reduced to the finally target in a
        // synchronized operation. We will cache over some serial inner loops,
        // if reduction is invariant to this loop, or if the loop densly
        // iterates over the reduction
        ID loopToCache; // Scope to flush locally accumulated result to target
                        // tensor
        std::vector<bool> preserveDim(op->indices_.size(), false);
        if (serialOverRed_.count(op->id())) {
            // Cache at out of the outer-most serial fully reduction loop
            for (auto &&loop : serialOverRed_.at(op->id())) {
                bool isReductionLoop = true;
                for (auto &&idx : _op->indices_) {
                    if (isVariant(variantMap_, {idx, op}, loop->id())) {
                        if (isDenseOver(idx, loop->iter_)) {
                            // is spatial loop
                            isReductionLoop = false;
                        } else {
                            // is randomly reduction loop
                            goto stop;
                        }
                    }
                }
                if (isReductionLoop) {
                    loopToCache = loop->id();
                }
            }
        stop:
            if (loopToCache.isValid()) {
                for (auto &&loop : serialOverRed_.at(op->id())) {
                    for (auto &&[idx, preserve] :
                         views::zip(_op->indices_, preserveDim)) {
                        if (isVariant(variantMap_, {idx, op}, loop->id()) &&
                            isDenseOver(idx, loop->iter_)) {
                            // is spatial loop
                            preserve = true;
                        }
                    }
                    if (loop->id() == loopToCache) {
                        break;
                    }
                }
            }
        }
        if (loopToCache.isValid()) {
            std::vector<Expr> newShape, newTargetIndices, newCacheIndices;
            for (auto &&[preserve, idx, dim] :
                 views::zip(preserveDim, _op->indices_,
                            buffer(_op->var_)->tensor()->shape())) {
                if (preserve) {
                    newCacheIndices.emplace_back(idx);
                    newShape.emplace_back(dim);
                    newTargetIndices.emplace_back(nullptr);
                } else {
                    newTargetIndices.emplace_back(idx);
                }
            }
            // Try to reuse existing cache array with the same size and the same
            // target indices
            for (auto &existing : cacheSync_[loopToCache]) {
                if (existing.oldNode_->var_ == _op->var_ &&
                    existing.preserveDim_ == preserveDim &&
                    ranges::equal(existing.newTargetIndices_, newTargetIndices,
                                  HashComparator{})) {
                    op->var_ +=
                        ".sync_cache." + toString(existing.oldNode_->id());
                    op->indices_ = std::move(newCacheIndices);
                    existing.isRandomAccess_ |= isRandomAccess;
                    goto done;
                }
            }
            cacheSync_[loopToCache].emplace_back(
                _op /* use the old name here */, newShape, newTargetIndices,
                isRandomAccess, preserveDim);
            op->var_ += ".sync_cache." + toString(op->id());
            op->indices_ = std::move(newCacheIndices);
        done:;
        } else {
            op->sync_ = true;
        }
    }
    return op;
}

Stmt MakeSyncReduction::visit(const For &_op) {
    auto oldGpuThreadDim = gpuThreadDim_;
    if (std::holds_alternative<CUDAScope>(_op->property_->parallel_) &&
        std::get<CUDAScope>(_op->property_->parallel_).level_ ==
            CUDAScope::Thread &&
        _op->len_->nodeType() == ASTNodeType::IntConst) {
        gpuThreadDim_ *= _op->len_.as<IntConstNode>()->val_;
    }
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    gpuThreadDim_ = oldGpuThreadDim;

    if (cacheSync_.count(op->id())) {
        Stmt ret = op;
        for (auto &&[reduce, newShape, targetIndices, isRandomAccess, _] :
             cacheSync_.at(op->id())) {
            auto cacheName =
                reduce->var_ + ".sync_cache." + toString(reduce->id());
            auto dtype = buffer(reduce->var_)->tensor()->dtype();
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
                cacheIndices, views::repeat(makeIntConst(0)), newShape,
                views::repeat(makeIntConst(1)), newShape,
                views::repeat(Ref<ForProperty>::make()), init);
            flush = makeNestedLoops(
                cacheIndices, views::repeat(makeIntConst(0)), newShape,
                views::repeat(makeIntConst(1)), newShape,
                views::repeat(Ref<ForProperty>::make()), flush);
            auto mtype = localMType(buffer(reduce->var_)->mtype(), dtype,
                                    newShape, isRandomAccess);
            ret = makeVarDef(cacheName,
                             makeBuffer(makeTensor(newShape, dtype),
                                        AccessType::Cache, mtype),
                             std::nullopt, makeStmtSeq({init, ret, flush}),
                             false);
        }
        return ret;
    } else {
        return op;
    }
}

Stmt makeParallelReduction(const Stmt &_op, const Ref<Target> &target) {
    auto op = makeReduction(_op);
    op = constFold(op); // For loop lengths

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
    auto found = [&](const Dependence &d) {
        ASSERT(d.dir_.size() >= 1);
        ASSERT(d.dir_.front().first.isNode_);
        auto &&loopId = d.dir_.front().first.id_;
        if (auto &&parallel = paraInfo.at(loopId).type_;
            std::holds_alternative<CUDAScope>(parallel) &&
            std::get<CUDAScope>(parallel).level_ == CUDAScope::Thread &&
            d.later() != d.earlier()) {
            // Use `__syncthreads` inserted by `pass/gpu/make_sync`, instead of
            // synchronizing individual `ReduceTo`s
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

    MakeLoopCarriedReduction makeLoopCarriedReduction(toAlter, variantExprMap);
    op = makeLoopCarriedReduction(op);
    op = MakeSyncReduction(makeLoopCarriedReduction.toUseSync(),
                           serialFinder.results(), variantExprMap, target)(op);
    op = simplify(op);
    return op;
}

} // namespace freetensor
