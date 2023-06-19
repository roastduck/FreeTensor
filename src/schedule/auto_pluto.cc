#include <unordered_set>

#include <analyze/all_uses.h>
#include <container_utils.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoPluto(const Ref<Target> &target) {
    // Try to transform
    //
    // for i = 0 to n
    //   a[i] += f(i)
    //   a[i + 1] += g(i)
    //
    // to
    //
    // for i = 0 to n + 1
    //   if i <= n
    //     a[i] += f(i)
    //   if i > 0
    //     a[i] += g(i - 1)
    //
    // by first `fission` and then `pluto_fuse` the loop. Currently, our
    // approach is to try such schedules only if they do not result in larger
    // intermediate tensors that would require additional memory usage.

    std::unordered_set<ID> tried;
    std::function<void(For nest)> tryLoop = [&, this](For nest) {
        Stmt lastSplitter;
        while (true) {
            std::string pattern; // Select the nest splitter in this loop body
            if (!lastSplitter.isValid()) {
                pattern =
                    "(<Store>|<ReduceTo>|<Eval>)<<-" + toString(nest->id());
            } else {
                pattern = "(<Store>|<ReduceTo>|<Eval>)&(<<-" +
                          toString(nest->id()) + ")&(:>>" +
                          toString(lastSplitter->id()) + ")";
            }
            Stmt splitter;
            if (auto &&allNext = findAll(pattern); !allNext.empty()) {
                splitter = allNext.front();
            }
            if (!splitter.isValid()) {
                break;
            }

            if (lastSplitter.isValid() && !tried.count(splitter->id())) {
                auto &&filterBefore =
                    parseSelector("<<:" + toString(splitter->id()));
                auto &&filterAfter =
                    parseSelector("!(<<:" + toString(splitter->id()) + ")");
                if (hasIntersect(allWrites(filterBefore, nest->body_),
                                 allUses(filterAfter, nest->body_)) ||
                    hasIntersect(allUses(filterBefore, nest->body_),
                                 allWrites(filterAfter, nest->body_))) {
                    beginTransaction();
                    try {
                        auto &&[frontMap, backMap] =
                            fission(nest->id(), FissionSide::Before,
                                    splitter->id(), false);
                        ID frontLoopId = frontMap.at(nest->id());
                        ID backLoopId = backMap.at(nest->id());
                        plutoFuse(frontLoopId, backLoopId);
                        commitTransaction();
                        tried.emplace(backMap.at(splitter->id()));
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                    }
                }
            }

            lastSplitter = splitter;
        }

        for (auto &&subNest :
             findAll("<For><-(!<For><-)*" + toString(nest->id()))) {
            tryLoop(subNest.as<ForNode>());
        }
    };
    for (auto &&loop : findAll("<For><-(!<For><-)*<-|")) {
        tryLoop(loop.as<ForNode>()); // Recurse into nested loops
    }
}

} // namespace freetensor
