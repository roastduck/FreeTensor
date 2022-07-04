#include <analyze/all_defs.h>
#include <analyze/all_no_reuse_defs.h>
#include <analyze/deps.h>
#include <pass/make_reduction.h>

namespace freetensor {

std::vector<ID> allNoReuseDefs(const Stmt &_op,
                               const std::unordered_set<AccessType> &atypes) {
    auto op = makeReduction(_op);
    std::unordered_set<ID> reusing;
    auto found = [&](const Dependency &d) { reusing.insert(d.defId()); };
    FindDeps().type(DEP_WAR).eraseOutsideVarDef(false)(op, found);
    std::vector<ID> ret;
    for (auto &&[id, name] : allDefs(op, atypes)) {
        if (!reusing.count(id)) {
            ret.emplace_back(id);
        }
    }
    return ret;
}

} // namespace freetensor
