#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/prop_one_time_use.h>
#include <pass/replace_uses.h>

namespace ir {

static bool sameParent(const Cursor &x, const Cursor &y) {
    if (!x.hasOuterCtrlFlow() && !y.hasOuterCtrlFlow()) {
        return true;
    }
    if (x.hasOuterCtrlFlow() && y.hasOuterCtrlFlow() &&
        x.outerCtrlFlow().id() == y.outerCtrlFlow().id()) {
        return true;
    }
    return false;
}

Stmt propOneTimeUse(const Stmt &op) {
    std::unordered_map<AST, std::vector<Stmt>> r2w, r2wMay;
    std::unordered_map<Stmt, std::vector<AST>> w2rMay;
    auto foundMay = [&](const Dependency &d) {
        r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
        w2rMay[d.earlier().as<StmtNode>()].emplace_back(d.later());
    };
    auto filterMust = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
        if (earlier.op_->nodeType() != ASTNodeType::Store) {
            return false;
        }
        if (earlier.def_->buffer_->atype() != AccessType::Cache) {
            return false;
        }
        if (!sameParent(later.cursor_, earlier.cursor_)) {
            // Definition of each vars may differ
            return false;
        }
        if (!r2wMay.count(later.op_) || r2wMay.at(later.op_).size() > 1 ||
            r2wMay.at(later.op_)[0] != earlier.op_.as<StmtNode>()) {
            return false;
        }
        if (!w2rMay.count(earlier.op_.as<StmtNode>()) ||
            w2rMay.at(earlier.op_.as<StmtNode>()).size() > 1 ||
            w2rMay.at(earlier.op_.as<StmtNode>())[0] != later.op_) {
            return false;
        }
        return true;
    };
    auto foundMust = [&](const Dependency &d) {
        if (checkNotModified(
                op, d.earlier().as<StoreNode>()->expr_,
                CheckNotModifiedSide::After, d.earlier_.cursor_.id(),
                CheckNotModifiedSide::Before, d.later_.cursor_.id())) {
            r2w[d.later()].emplace_back(d.earlier().as<StmtNode>());
        }
    };
    findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW, nullptr, false);
    findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW, filterMust);

    std::unordered_map<Load, Expr> replaceLoad;
    std::unordered_map<ReduceTo, Expr> replaceReduceTo;
    for (auto &&item : r2w) {
        ASSERT(item.second.size() == 1);
        ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
        auto &&store = item.second.front().as<StoreNode>();
        if (item.first->nodeType() == ASTNodeType::Load) {
            auto &&load = item.first.as<LoadNode>();
            replaceLoad[load] = store->expr_;
        } else {
            ASSERT(item.first->nodeType() == ASTNodeType::ReduceTo);
            auto &&reduceTo = item.first.as<ReduceToNode>();
            replaceReduceTo[reduceTo] = store->expr_;
        }
    }

    return ReplaceUses(replaceLoad, replaceReduceTo)(op);
}

} // namespace ir

