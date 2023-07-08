#include <unordered_set>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
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
        auto triedWhenBegin = tried;
        beginTransaction();

        try {
            std::vector<ID>
                splitterIds; // Record IDs because we are mutating ast()
            for (auto &&splitter : findAll("(<Store>|<ReduceTo>|<Eval>)<<-" +
                                           toString(nest->id()))) {
                splitterIds.emplace_back(splitter->id());
            }

            std::vector<ID> parts;
            for (size_t i = 1, n = splitterIds.size(); i < n; i++) {
                auto &&splitterId = splitterIds[i];
                if (!tried.count(splitterId)) {
                    // Check access in the first part of the body. (No need to
                    // check spliterIds[i - 1] because already fissioned or
                    // failed)
                    auto &&filterBefore =
                        parseSelector("<<:" + toString(splitterId));
                    // Check access in the second part of the body.
                    auto &&filterAfter = parseSelector(
                        "!(<<:" + toString(splitterId) + ")" +
                        (i + 1 < n
                             ? "&(<<:" + toString(splitterIds[i + 1]) + ")"
                             : ""));

                    // Coarse filter: same variable accessed in both parts
                    if (!hasIntersect(allWrites(filterBefore, nest->body_),
                                      allUses(filterAfter, nest->body_)) &&
                        !hasIntersect(allUses(filterBefore, nest->body_),
                                      allWrites(filterAfter, nest->body_))) {
                        continue;
                    }

                    // Fine filter: dependence across the splitter
                    auto stmtsBefore = findAll(filterBefore);
                    auto stmtsAfter = findAll(filterAfter);
                    if (!FindDeps()
                             // Don't restrict `.direction` to `nest->id()`,
                             // beaucse inner loops matters
                             .ignoreReductionWAW(false)
                             .filter([&](auto &&later, auto &&earlier) {
                                 return (std::ranges::count(stmtsBefore,
                                                            later.stmt_) &&
                                         std::ranges::count(stmtsAfter,
                                                            earlier.stmt_)) ||
                                        (std::ranges::count(stmtsAfter,
                                                            later.stmt_) &&
                                         std::ranges::count(stmtsBefore,
                                                            earlier.stmt_));
                             })
                             .exists(ast())) {
                        continue;
                    }

                    try {
                        auto &&[frontMap, backMap] = fission(
                            nest->id(), FissionSide::Before, splitterId, false,
                            "." + std::to_string(parts.size()), "");
                        ID frontLoopId = frontMap.at(nest->id());
                        ID backLoopId = backMap.at(nest->id());
                        if (!parts.empty()) {
                            parts.pop_back();
                        }
                        parts.emplace_back(frontLoopId);
                        parts.emplace_back(backLoopId);
                        tried.emplace(backMap.at(splitterId));
                        nest = find(backLoopId).as<ForNode>();
                    } catch (const InvalidSchedule &e) {
                        // Do nothing
                    }
                }
            }

            if (parts.size() > 1) {
                ID fusedId = parts[0];
                for (size_t i = 1, n = parts.size(); i < n; i++) {
                    fusedId = plutoFuse(fusedId, parts[i]).first;
                }
            }

            commitTransaction();
        } catch (const InvalidSchedule &e) {
            abortTransaction();
            tried = std::move(triedWhenBegin);
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
