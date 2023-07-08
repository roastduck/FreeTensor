#include <analyze/deps.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoReorder(const Ref<Target> &target) {
    auto allLoops = findAll("<For>");
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop->id(), DepDirection::Normal}});
    }

    // 0 = No dep
    // 1 = Reduction
    // 2 = Others
    std::unordered_map<ID, int> depLevel;
    FindDeps().direction(direction).ignoreReductionWAW(false)(
        ast(), [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            auto &level = depLevel[d.dir_[0].first.id_];
            if (d.earlier()->nodeType() == ASTNodeType::ReduceTo &&
                d.later()->nodeType() == ASTNodeType::ReduceTo) {
                level = std::max(level, 1);
            } else {
                level = std::max(level, 2);
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
