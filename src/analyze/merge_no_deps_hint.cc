#include <algorithm>

#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <analyze/merge_no_deps_hint.h>
#include <container_utils.h>

namespace freetensor {

std::vector<std::string> mergeNoDepsHint(const Stmt &ast,
                                         const std::vector<ID> &loops) {
    std::vector<For> loopNodes;
    loopNodes.reserve(loops.size());
    for (auto &&loopId : loops) {
        auto node = findStmt(ast, loopId);
        ASSERT(node->nodeType() == ASTNodeType::For);
        loopNodes.emplace_back(node.as<ForNode>());
    }

    std::vector<std::string> ret, candidates;
    for (auto &&[i, loop] : views::enumerate(loopNodes)) {
        if (i == 0) {
            ret = loop->property_->noDeps_;
            candidates = loop->property_->noDeps_;
        } else {
            ret = intersect(ret, loop->property_->noDeps_);
            candidates = uni(candidates, loop->property_->noDeps_);
        }
    }

    for (auto &&item : candidates) {
        if (std::find(ret.begin(), ret.end(), item) == ret.end()) {
            for (auto &&loop : loopNodes) {
                if (std::find(loop->property_->noDeps_.begin(),
                              loop->property_->noDeps_.end(),
                              item) == loop->property_->noDeps_.end()) {
                    bool noDep = true;
                    auto found = [&](const Dependence &d) { noDep = false; };
                    FindDeps()
                        .direction({{{loop->id(), DepDirection::Different}}})
                        .filterAccess([&](const auto &acc) {
                            return acc.def_->name_ == item;
                        })
                        .eraseOutsideVarDef(false)
                        .ignoreReductionWAW(false)(ast, found);
                    if (!noDep) {
                        goto fail;
                    }
                }
            }
            ret.emplace_back(item);
        fail:;
        }
    }
    return ret;
}

} // namespace freetensor
