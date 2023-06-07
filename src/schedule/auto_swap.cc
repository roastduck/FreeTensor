#include <analyze/deps.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoSwap(const Ref<Target> &target) {
    // If a statement S1 always happens EARLIER THAN another statement S2 with
    // respect to the item they write to, S1 should be put ABOVE S2 in program
    // order if possible, so as to enable more fission and fusion. For example,
    //
    // for i = 0 to n {
    //   a[i] += f(i)  // -------- (S1)
    //   a[i + 1] = g(i)  // ----- (S2)
    // }
    //
    // S2 always happens earlier than S1, so S1 and S2 should be swapped to
    // enable fissioning the loop.

    std::unordered_set<std::pair<ID, ID>>
        candidates; // (earlier or lower, later or upper)
    FindDeps().type(DEP_WAW).filter(
        [](const AccessPoint &later, const AccessPoint &earlier) {
            // later is above ealier in program order
            return later.stmt_->isBefore(earlier.stmt_);
        })(ast(), [&](const Dependence &d) {
        logger() << "Found (1) " << d << std::endl;
        // Lower-to-upper (reversed) dependence is found
        candidates.emplace(d.earlier_.stmt_->id(), d.later_.stmt_->id());
    });

    FindDeps().type(DEP_WAW).ignoreReductionWAW(false).filter(
        [&](const AccessPoint &later, const AccessPoint &earlier) {
            // Check for upper-to-lower (normal) dependence
            return candidates.count(
                std::make_pair(later.stmt_->id(), earlier.stmt_->id()));
        })(ast(), [&](const Dependence &d) {
        logger() << "Found (2) " << d << std::endl;
        // Don't swap if there are dependeces in both directions
        candidates.erase(
            std::make_pair(d.later_.stmt_->id(), d.earlier_.stmt_->id()));
    });

    for (auto &&[earlierId, laterId] : candidates) {
        try {
            moveTo(laterId, MoveToSide::After, earlierId);
        } catch (const InvalidSchedule &e) {
            logger() << "error " << e.what() << std::endl;
            try {
                moveTo(earlierId, MoveToSide::Before, laterId);
            } catch (const InvalidSchedule &e) {
                // give up
            }
        }
    }
}

} // namespace freetensor
