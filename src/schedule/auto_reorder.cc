#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <lazy.h>
#include <schedule.h>

namespace freetensor {

static bool isRandomReduce(const auto &variants, const ReduceTo &reduce,
                           const ID &loopId) {
    // a[i] += ...       --> No dependence, already ruled out
    // a[idx[i]] += ...  --> Loop variant, random reduction
    // a[] += ...        --> Loop invariant, loop-carried reduction
    for (auto &&idx : reduce->indices_) {
        if (isVariant(variants, StmtOrExprID{idx, reduce}, loopId)) {
            return true;
        }
    }
    return false;
}

void Schedule::autoReorder(const Ref<Target> &target) {
    auto allLoops = findAll("<For>");
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop->id(), DepDirection::Normal}});
    }

    // Sort loops according to levels
    [[maybe_unused]] int levelNoDep =
        0; // std::unordered_map automatically initialize values to 0
    int levelRandomReduction, levelLoopCarriedReduction, levelOthers;
    switch (target->type()) {
    case TargetType::CPU:
        // Loop-carried parallel reduction on CPUs is very efficient: just
        // reduce to core-local accumulators. There will be no more accumulators
        // than visible cores to the OS. On the contrary, reordering these loops
        // may lead to inefficient memory layout. Thus we treat reductions the
        // same is dependence-free loops.
        //
        // TODO: For random reductions, if there are a random reduction and a
        // loop-carried reduction nested together, it depends on whether they
        // are reducing into the same variable. If they are not the same, we
        // should reorder the random one inside the avoid atomic-operator
        // overhead. Otherwise, don't reorder it to prevent enlarging the
        // accumulator.
        levelLoopCarriedReduction = 0;
        levelRandomReduction = 1;
        levelOthers = 1;
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        // Reduction on GPUs comes with high overhead, so place them inside
        // dependence-free loops. Loop-carried reductions performs well only
        // within one block, while random reduction can perform across blocks,
        // so reorder loop-carried reductions inside random reductions.
        levelRandomReduction = 1;
        levelLoopCarriedReduction = 2;
        levelOthers = 3;
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }

    std::unordered_map<ID, int> depLevel;
    auto variants = LAZY(findLoopVariance(ast()).first);
    FindDeps().direction(direction).ignoreReductionWAW(false)(
        ast(), [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            auto &&loopId = d.dir_[0].first.id_;
            auto &level = depLevel[loopId];
            if (d.earlier()->nodeType() == ASTNodeType::ReduceTo &&
                d.later()->nodeType() == ASTNodeType::ReduceTo) {
                if (isRandomReduce(*variants, d.earlier().as<ReduceToNode>(),
                                   loopId) ||
                    isRandomReduce(*variants, d.later().as<ReduceToNode>(),
                                   loopId)) {
                    level = std::max(level, levelRandomReduction);
                } else {
                    level = std::max(level, levelLoopCarriedReduction);
                }
            } else {
                level = std::max(level, levelOthers);
            }
        });

    std::function<void(For nest)> visitNest = [&, this](For nest) {
        // Currently we only reorder loops in a perfect loop nest
        std::vector<ID> perfectNest = {nest->id()};
        while (true) {
            if (auto inners =
                    findAll("<For><-(!<For><-)*" + toString(nest->id()));
                inners.size() == 1) {
                nest = inners.front().as<ForNode>();
                perfectNest.emplace_back(nest->id());
            } else {
                break;
            }
        }

        ID innerMost = perfectNest.back();
        while (perfectNest.size() > 1) {
            auto sorted = perfectNest;
            std::stable_sort(sorted.begin(), sorted.end(),
                             [&](const ID &lhs, const ID &rhs) {
                                 return depLevel[lhs] < depLevel[rhs];
                             });
            if (sorted == perfectNest) {
                break;
            }
            try {
                reorder(sorted, ReorderMode::MoveOutImperfect);
                innerMost = sorted.back();
                break;
            } catch (const InvalidSchedule &e) {
                // Retry with one less loop
                perfectNest.pop_back();
                innerMost = perfectNest.back();
            }
        }

        for (auto &&subNest :
             findAll("<For><-(!<For><-)*" + toString(innerMost))) {
            visitNest(subNest.as<ForNode>());
        }
    };
    for (auto &&subNest : findAll("<For><-(!<For><-)*<-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        visitNest(subNest.as<ForNode>());
    }
}

} // namespace freetensor
