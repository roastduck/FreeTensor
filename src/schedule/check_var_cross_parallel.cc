#include <analyze/deps.h>
#include <schedule/check_var_cross_parallel.h>

namespace ir {

void checkVarCrossParallel(const Stmt &ast, const std::string &def,
                           MemType mtype) {
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.def_->id() == def;
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        throw InvalidSchedule(
            dep2Str(d.cond_[0].first, d.var_, d.later(), d.earlier()));
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

