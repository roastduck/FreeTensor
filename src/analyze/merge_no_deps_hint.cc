#include <algorithm>

#include <analyze/deps.h>
#include <analyze/merge_no_deps_hint.h>
#include <analyze/with_cursor.h>
#include <container_utils.h>

namespace ir {

std::vector<std::string> mergeNoDepsHint(const Stmt &ast, const ID &loop1,
                                         const ID &loop2) {
    auto _l1 = getCursorById(ast, loop1).node();
    ASSERT(_l1->nodeType() == ASTNodeType::For);
    auto l1 = _l1.as<ForNode>();
    auto _l2 = getCursorById(ast, loop2).node();
    ASSERT(_l2->nodeType() == ASTNodeType::For);
    auto l2 = _l2.as<ForNode>();

    auto ret = intersect(l1->property_.noDeps_, l2->property_.noDeps_);

    for (auto &&item : l1->property_.noDeps_) {
        if (std::find(ret.begin(), ret.end(), item) == ret.end()) {
            bool noDep = true;
            auto filter = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
                return later.def_->name_ == item;
            };
            auto found = [&](const Dependency &d) { noDep = false; };
            findDeps(ast, {{{loop2, DepDirection::Different}}}, found,
                     FindDepsMode::Dep, DEP_ALL, filter, false, false);
            if (noDep) {
                ret.emplace_back(item);
            }
        }
    }

    for (auto &&item : l2->property_.noDeps_) {
        if (std::find(ret.begin(), ret.end(), item) == ret.end()) {
            bool noDep = true;
            auto filter = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
                return later.def_->name_ == item;
            };
            auto found = [&](const Dependency &d) { noDep = false; };
            findDeps(ast, {{{loop1, DepDirection::Different}}}, found,
                     FindDepsMode::Dep, DEP_ALL, filter, false, false);
            if (noDep) {
                ret.emplace_back(item);
            }
        }
    }

    return ret;
}

} // namespace ir
