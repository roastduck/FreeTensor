#include <analyze/all_defs.h>
#include <analyze/find_stmt.h>
#include <autograd/all_no_reuse_defs.h>
#include <autograd/tape_strategy.h>
#include <container_utils.h>

namespace freetensor {

std::unordered_set<ID> TapeStrategy::getIdsToTape(const Stmt &ast) const {
    std::unordered_set<ID> ret;

    switch (mode_) {
    case GradTapeMode::All: {
        auto all = allDefs(
            ast, {AccessType::Cache, AccessType::Output, AccessType::InOut});
        ret = ranges::to<std::unordered_set>(all | views::keys);
        break;
    }
    case GradTapeMode::Nothing: {
        // All InOut vars must be taped
        auto inouts = allDefs(ast, {AccessType::InOut});
        ret = ranges::to<std::unordered_set>(inouts | views::keys);
        break;
    }
    case GradTapeMode::NoReuseOnly: {
        auto noReuse = ranges::to<std::unordered_set>(
            allNoReuseDefs(ast, {AccessType::Cache, AccessType::Output}));
        // All InOut vars must be taped
        auto inouts = allDefs(ast, {AccessType::InOut});
        ret =
            uni(noReuse, ranges::to<std::unordered_set>(inouts | views::keys));
        break;
    }
    default:
        ASSERT(false);
    }

    for (auto &&item : alwaysTape_) {
        if (std::holds_alternative<ID>(item)) {
            ret.emplace(std::get<ID>(item));
        } else if (std::holds_alternative<std::string>(item)) {
            ret.emplace(findStmt(ast, std::get<std::string>(item))->id());
        } else {
            ret.emplace(findStmt(ast, std::get<Ref<Selector>>(item))->id());
        }
    }
    for (auto &&item : neverTape_) {
        if (std::holds_alternative<ID>(item)) {
            ret.erase(std::get<ID>(item));
        } else if (std::holds_alternative<std::string>(item)) {
            ret.erase(findStmt(ast, std::get<std::string>(item))->id());
        } else {
            ret.erase(findStmt(ast, std::get<Ref<Selector>>(item))->id());
        }
    }
    return ret;
}

} // namespace freetensor
