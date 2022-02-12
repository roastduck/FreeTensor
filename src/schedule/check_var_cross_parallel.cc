#include <analyze/deps.h>
#include <schedule/check_var_cross_parallel.h>

namespace ir {

void checkVarCrossParallel(const Stmt &ast, const ID &def, MemType mtype) {
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.def_->id() == def;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    std::vector<FindDepsCond> conds;
    switch (mtype) {
    case MemType::GPULocal:
        conds.push_back({{NodeIDOrParallelScope("threadIdx.x", false),
                          DepDirection::Different}});
        conds.push_back({{NodeIDOrParallelScope("threadIdx.y", false),
                          DepDirection::Different}});
        conds.push_back({{NodeIDOrParallelScope("threadIdx.z", false),
                          DepDirection::Different}});
        // fall through
    case MemType::GPUShared:
        conds.push_back({{NodeIDOrParallelScope("blockIdx.x", false),
                          DepDirection::Different}});
        conds.push_back({{NodeIDOrParallelScope("blockIdx.y", false),
                          DepDirection::Different}});
        conds.push_back({{NodeIDOrParallelScope("blockIdx.z", false),
                          DepDirection::Different}});
        break;
    default:; // do nothing
    }
    findDeps(ast, conds, found, FindDepsMode::Dep, DEP_ALL, filter);
}

} // namespace ir

