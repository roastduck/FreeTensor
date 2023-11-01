#include <analyze/all_defs.h>
#include <analyze/deps.h>
#include <autograd/all_no_reuse_defs.h>
#include <pass/make_reduction.h>

namespace freetensor {

std::vector<ID> allNoReuseDefs(const Stmt &_op,
                               const std::unordered_set<AccessType> &atypes) {
    auto op = makeReduction(_op);
    std::vector<ID> ret;
    for (auto &&[id, name] : allDefs(op, atypes)) {
        std::vector<FindDepsDir> direction;
        // Check for all scopes outer of this variable
        for (auto &&scope : findAllStmt(op, "<For>->>" + toString(id))) {
            // TODO: Also check for <StmtSeq> scopes. Currently we don't check
            // it because analyze/deps merges StmtSeq scopes, which makes us
            // unable to distinguish StmtSeq out of the VarDef or inside the
            // VarDef
            direction.push_back({{scope->id(), DepDirection::Normal}});
        }
        if (!FindDeps()
                 .type(DEP_WAR | DEP_RAW)
                 .direction(direction)
                 .eraseOutsideVarDef(false)
                 .exists(op)) {
            ret.emplace_back(id);
        }
    }
    return ret;
}

} // namespace freetensor
