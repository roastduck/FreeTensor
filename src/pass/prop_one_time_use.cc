#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/prop_one_time_use.h>
#include <pass/replace_uses.h>
#include <pass/sink_var.h>

namespace ir {

static bool sameParent(const Stmt &x, const Stmt &y) {
    return x->parentCtrlFlow() == y->parentCtrlFlow();
}

Stmt propOneTimeUse(const Stmt &_op) {
    auto op = makeReduction(_op);

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call ReplaceUses to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    op = hoistVarOverStmtSeq(op);

    std::unordered_map<AST, std::vector<Stmt>> r2w, r2wMay;
    std::unordered_map<Stmt, std::vector<AST>> w2r, w2rMay;
    auto filterMust = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
        if (earlier.op_->nodeType() != ASTNodeType::Store) {
            return false;
        }
        if (earlier.def_->buffer_->atype() != AccessType::Cache) {
            return false;
        }
        if (later.op_->nodeType() == ASTNodeType::ReduceTo) {
            return false; // pass/remove_write will deal with it
        }
        if (!sameParent(later.stmt_, earlier.stmt_)) {
            // Definition of each vars may differ
            return false;
        }
        return true;
    };
    auto foundMust = [&](const Dependency &d) {
        if (checkNotModified(
                op, d.earlier().as<StoreNode>()->expr_,
                CheckNotModifiedSide::Before, d.earlier_.stmt_->id(),
                CheckNotModifiedSide::Before, d.later_.stmt_->id())) {
            r2w[d.later()].emplace_back(d.earlier().as<StmtNode>());
            w2r[d.earlier().as<StmtNode>()].emplace_back(d.later());
        }
    };
    auto filterMay = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return r2w.count(later.op_) || w2r.count(earlier.op_.as<StmtNode>());
    };
    auto foundMay = [&](const Dependency &d) {
        r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
        w2rMay[d.earlier().as<StmtNode>()].emplace_back(d.later());
    };
    findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW, filterMust);
    findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW, filterMay, false);

    std::unordered_map<AST, Expr> replace;
    for (auto &&item : r2w) {
        if (item.second.size() > 1) {
            continue;
        }
        ASSERT(item.second.size() == 1);
        if (!r2wMay.count(item.first) || r2wMay.at(item.first).size() > 1 ||
            r2wMay.at(item.first)[0] != item.second.front()) {
            continue;
        }
        if (!w2rMay.count(item.second.front()) ||
            w2rMay.at(item.second.front()).size() > 1 ||
            w2rMay.at(item.second.front())[0] != item.first) {
            continue;
        }
        ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
        auto &&store = item.second.front().as<StoreNode>();
        replace[item.first] = store->expr_;
    }

    op = ReplaceUses(replace)(op);
    return sinkVar(op);
}

} // namespace ir
