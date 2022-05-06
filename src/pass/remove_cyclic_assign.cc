#include <analyze/deps.h>
#include <pass/remove_cyclic_assign.h>

namespace freetensor {

Stmt removeCyclicAssign(const Stmt &op) {
    std::unordered_map<Stmt, Stmt> later2earlier;
    std::unordered_set<Stmt> redundant;
    auto foundRAW = [&](const Dependency &d) {
        if (d.earlier()->nodeType() == ASTNodeType::Store) {
            if (auto &&laterStmt = d.later_.stmt_;
                laterStmt->nodeType() == ASTNodeType::Store) {
                if (auto &&laterStore = laterStmt.as<StoreNode>();
                    laterStore->expr_ == d.later()) {
                    later2earlier[d.later_.stmt_] = d.earlier_.stmt_;
                }
            }
        }
    };
    auto filterWAR = [&](const AccessPoint &later, const AccessPoint &earlier) {
        if (later2earlier.count(later.stmt_) &&
            later2earlier.at(later.stmt_) == earlier.stmt_) {
            auto &&earlierStmt = earlier.stmt_;
            ASSERT(earlierStmt->nodeType() == ASTNodeType::Store);
            if (auto &&earlierStore = earlierStmt.as<StoreNode>();
                earlierStore->expr_ == earlier.op_) {
                return true;
            }
        }
        return false;
    };
    auto foundWAR = [&](const Dependency &d) {
        redundant.emplace(d.later_.stmt_);
    };
    // No filter for RAW because we want findDeps to find the nearest affecting
    // write
    findDeps(op, {{}}, foundRAW, FindDepsMode::KillLater, DEP_RAW);
    findDeps(op, {{}}, foundWAR, FindDepsMode::KillLater, DEP_WAR, filterWAR);
    if (!redundant.empty()) {
        return RemoveWrites(redundant, {})(op);
    } else {
        return op;
    }
}

} // namespace freetensor
