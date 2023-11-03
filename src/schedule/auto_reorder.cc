#include <analyze/deps.h>
#include <pass/const_fold.h>
#include <schedule.h>

namespace freetensor {

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
    int levelReduction = 1;
    int levelOthers = 2;

    std::unordered_map<ID, int> depLevel;
    FindDeps().direction(direction).ignoreReductionWAW(false)(
        ast(), [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            auto &level = depLevel[d.dir_[0].first.id_];
            if (d.earlier()->nodeType() == ASTNodeType::ReduceTo &&
                d.later()->nodeType() == ASTNodeType::ReduceTo) {
                level = std::max(level, levelReduction);
            } else {
                level = std::max(level, levelOthers);
            }
        });

    // schedule/auto_reorder focuses on improving parallelizability, so we
    // only reorder parallelizable loops out, and make them parallelizable
    // for schedule/auto_parallelize. Reordering inner loops may lead to
    // inefficient memory layout, so we keep them.
    //
    // This is done by a selection sort. We select as many dependence-free
    // loops as possible, and select dependence-free loops plus reduction
    // loops up to a total length limit, and no other loops.
    int enoughParDgr;
    switch (target->type()) {
    case TargetType::CPU:
        enoughParDgr = target.as<CPUTarget>()->nCores();
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        int numSM = target.as<GPUTarget>()->multiProcessorCount();
        int maxThreads = 256; // Sync this number with auto_parallelize
        enoughParDgr = numSM * maxThreads;
        break;
    }
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }

    std::function<void(For nest)> visitNest = [&, this](For nest) {
        // Currently we only reorder loops in a perfect loop nest
        std::vector<ID> perfectNest = {nest->id()};
        std::unordered_map<ID, int64_t> constLen;
        while (true) {
            if (auto inners =
                    findAll("<For><-(!<For><-)*" + toString(nest->id()));
                inners.size() == 1) {
                nest = inners.front().as<ForNode>();
                perfectNest.emplace_back(nest->id());
                if (auto l = constFold(nest->len_);
                    l->nodeType() == ASTNodeType::IntConst) {
                    constLen[nest->id()] = l.as<IntConstNode>()->val_;
                }
            } else {
                break;
            }
        }

        ID innerMost = perfectNest.back();
        while (perfectNest.size() > 1) {
            // Selection sort
            auto sorted = perfectNest;
            int64_t parDgr = 1; // -1 = dynamic length
            for (size_t i = 0, n = perfectNest.size(); i < n; i++) {
                size_t best = i;
                auto bestId = sorted[i];
                for (size_t j = i + 1; j < n; j++) {
                    if (depLevel[sorted[j]] < depLevel[bestId]) {
                        best = j, bestId = sorted[j];
                    }
                }
                if (depLevel[bestId] >= levelOthers) {
                    break;
                }
                if (parDgr == -1 || parDgr >= enoughParDgr) {
                    break;
                }
                std::swap(sorted[i], sorted[best]);
                if (parDgr != -1) {
                    if (auto l = constLen.find(bestId); l != constLen.end()) {
                        parDgr = parDgr * l->second;
                    } else {
                        parDgr = -1;
                    }
                }
            }

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
