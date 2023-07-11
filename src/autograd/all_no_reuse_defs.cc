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
        for (auto &&scope :
             findAllStmt(op, "(<For>|<StmtSeq>)->" + toString(id))) {
            // NOTE: If checking each `StmtSeq` is too slow, we can check node
            // positions in the AST in the `found` callback
            direction.push_back({{scope->id(), DepDirection::Normal}});
        }
        if (!FindDeps()
                 .type(DEP_WAR)
                 .direction(direction)
                 .eraseOutsideVarDef(false)
                 .exists(op)) {
            ret.emplace_back(id);
        }
    }
    return ret;
}

} // namespace freetensor
