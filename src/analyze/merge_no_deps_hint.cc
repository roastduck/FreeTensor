#include <algorithm>

#include <itertools.hpp>

#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <analyze/merge_no_deps_hint.h>
#include <container_utils.h>

namespace ir {

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
    for (auto &&[i, loop] : iter::enumerate(loopNodes)) {
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
                    auto filter = [&](const AccessPoint &later,
                                      const AccessPoint &earlier) {
                        return later.def_->name_ == item;
                    };
                    auto found = [&](const Dependency &d) { noDep = false; };
                    findDeps(ast, {{{loop->id(), DepDirection::Different}}},
                             found, FindDepsMode::Dep, DEP_ALL, filter, false,
                             false);
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

} // namespace ir
