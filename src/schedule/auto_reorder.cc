#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoReorder(const Ref<Target> &target) {
    auto allLoops = findAllLoops(ast());
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop, DepDirection::Normal}});
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
                    findAll("<For><-(!<For><-)*#" + toString(nest->id()));
                inners.size() == 1) {
                nest = inners.front().as<ForNode>();
                perfectNest.emplace_back(nest->id());
            } else {
                break;
            }
        }

        auto sorted = perfectNest;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [&](const ID &lhs, const ID &rhs) {
                             return depLevel[lhs] < depLevel[rhs];
                         });
        if (sorted != perfectNest) {
            reorder(sorted);
        }

        for (auto &&subNest :
             findAll("<For><-(!<For><-)*#" + toString(nest->id()))) {
            visitNest(subNest.as<ForNode>());
        }
    };
    for (auto &&subNest : findAll("<For><-(!<For><-)*<-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        visitNest(subNest.as<ForNode>());
    }
}

} // namespace freetensor
