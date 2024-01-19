#ifndef FREE_TENSOR_ALL_DEFS_H
#define FREE_TENSOR_ALL_DEFS_H

#include <unordered_set>

#include <analyze/find_stmt.h>
#include <container_utils.h>

namespace freetensor {

/**
 * Collect IDs of all `VarDef` nodes of specific `AccessType`s
 */
inline std::vector<std::pair<ID, std::string>>
allDefs(const Stmt &op,
        const std::unordered_set<AccessType> &atypes =
            allAccessTypes | ranges::to<std::unordered_set>() {
    std::vector<std::pair<ID, std::string>> ret;
    for (auto &&node : findAllStmt(op, [&](const Stmt &s) {
             return s->nodeType() == ASTNodeType::VarDef &&
                    atypes.count(s.as<VarDefNode>()->buffer_->atype());
         })) {
        ret.emplace_back(node->id(), node.as<VarDefNode>()->name_);
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_ALL_DEFS_H
