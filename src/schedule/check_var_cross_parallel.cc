#include <analyze/deps.h>
#include <schedule/check_var_cross_parallel.h>

namespace freetensor {

void checkVarCrossParallel(const Stmt &ast, const ID &def, MemType mtype) {
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.def_->id() == def;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() == 1);
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    std::vector<FindDepsDir> direction;
    switch (mtype) {
    case MemType::GPULocal:
        direction.push_back(
            {{NodeIDOrParallelScope(threadIdxX), DepDirection::Different}});
        direction.push_back(
            {{NodeIDOrParallelScope(threadIdxY), DepDirection::Different}});
        direction.push_back(
            {{NodeIDOrParallelScope(threadIdxZ), DepDirection::Different}});
        // fall through
    case MemType::GPUWarp:
    case MemType::GPUShared:
        direction.push_back(
            {{NodeIDOrParallelScope(blockIdxX), DepDirection::Different}});
        direction.push_back(
            {{NodeIDOrParallelScope(blockIdxY), DepDirection::Different}});
        direction.push_back(
            {{NodeIDOrParallelScope(blockIdxZ), DepDirection::Different}});
        break;
    default:; // do nothing
    }
    FindDeps().direction(direction).filter(filter)(ast, found);
}

} // namespace freetensor
