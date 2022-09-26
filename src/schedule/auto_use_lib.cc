#include <analyze/all_defs.h>
#include <analyze/all_stmts.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoUseLib(const Target &target) {
    // Try to implement each top-level loops with lib calls
    for (auto &&_loop : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto loop = _loop.as<ForNode>();
        try {
            asMatMul(loop->id());
        } catch (const InvalidSchedule &e) {
            // If the loop is marked as preferLibs, we inline all local
            // variables, fission all the statments apart, and try applying to
            // each of them
            bool isPreferLibs = false;
            for (For l = loop;;) {
                if (l->property_->preferLibs_) {
                    isPreferLibs = true;
                    break;
                }
                Stmt body = l->body_;
                while (body->nodeType() == ASTNodeType::VarDef) {
                    body = body.as<VarDefNode>()->body_;
                }
                if (body->nodeType() != ASTNodeType::For) {
                    break;
                } else {
                    l = body.as<ForNode>();
                }
            }
            if (isPreferLibs) {
                for (auto &&[defId, name] :
                     allDefs(loop, {AccessType::Cache})) {
                    try {
                        inlining(defId);
                    } catch (const InvalidSchedule &e) {
                        // do nothing
                    }
                }
                auto stmts =
                    allStmts(loop, {ASTNodeType::Store, ASTNodeType::ReduceTo});
                for (auto &&[i, stmt] : views::enumerate(stmts)) {
                    beginTransaction();
                    try {
                        fission(loop->id(), FissionSide::Before, stmt->id(),
                                "." + toString(i), "");
                        auto libStmtId =
                            fission(loop->id(), FissionSide::After, stmt->id(),
                                    "." + toString(i) + ".lib", "")
                                .first.at(loop->id());
                        asMatMul(libStmtId);
                        commitTransaction();
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                    }
                }
            }
        }
    }
}

} // namespace freetensor
