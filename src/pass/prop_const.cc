#include <analyze/all_iters.h>
#include <analyze/all_reads.h>
#include <analyze/deps.h>
#include <pass/prop_const.h>
#include <pass/replace_uses.h>
#include <pass/simplify.h>

namespace ir {

static bool iterDefined(Cursor c, const std::string &name) {
    while (true) {
        if (c.nodeType() == ASTNodeType::For &&
            c.node().as<ForNode>()->iter_ == name) {
            return true;
        }
        if (!c.hasOuter()) {
            return false;
        }
        c = c.outer();
    }
}

Stmt propConst(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = simplifyPass(op);

        std::unordered_map<AST, std::vector<Stmt>> r2w, r2wMay;
        auto filterMust = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
            if (earlier.op_->nodeType() != ASTNodeType::Store) {
                return false;
            }
            if (later.op_->nodeType() == ASTNodeType::ReduceTo) {
                return false; // pass/remove_write will deal with it
            }
            auto &&expr = earlier.op_.as<StoreNode>()->expr_;
            if (!allReads(expr).empty()) {
                // Expressions should contain only constants and iterating vars
                return false;
            }
            auto &&iters = allIters(expr);
            auto common = lca(later.cursor_, earlier.cursor_);
            for (auto &&iter : iters) {
                if (!iterDefined(common, iter)) {
                    // Iter with the same name from different loops
                    return false;
                }
            }
            return true;
        };
        auto foundMust = [&](const Dependency &d) {
            auto &&expr = d.earlier().as<StoreNode>()->expr_;
            auto &&iters = allIters(expr);
            auto common = lca(d.later_.cursor_, d.earlier_.cursor_);
            auto dep = d.dep_;
            for (auto &&iter : iters) {
                for (auto c = common; c.isValid(); c = c.outer()) {
                    if (c.nodeType() == ASTNodeType::For) {
                        if (auto &&f = c.node().as<ForNode>();
                            f->iter_ == iter) {
                            dep =
                                d.extraCheck(dep, f->id(), DepDirection::Same);
                            if (dep != d.dep_) {
                                // Iterating variable in different iterations
                                return;
                            }
                            break;
                        }
                    }
                }
            }
            r2w[d.later()].emplace_back(d.earlier().as<StmtNode>());
        };
        auto filterMay = [&](const AccessPoint &later,
                             const AccessPoint &earlier) {
            return r2w.count(later.op_);
        };
        auto foundMay = [&](const Dependency &d) {
            r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
        };
        findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW,
                 filterMust);
        findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW, filterMay,
                 false);

        std::unordered_map<Load, Expr> replaceLoad;
        for (auto &&item : r2w) {
            if (item.second.size() > 1) {
                continue;
            }
            ASSERT(item.second.size() == 1);
            if (!r2wMay.count(item.first) || r2wMay.at(item.first).size() > 1 ||
                r2wMay.at(item.first)[0] != item.second.front()) {
                continue;
            }
            ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
            auto &&store = item.second.front().as<StoreNode>();
            ASSERT(item.first->nodeType() == ASTNodeType::Load);
            auto &&load = item.first.as<LoadNode>();
            replaceLoad[load] = store->expr_;
        }

        if (replaceLoad.empty() || i > 100) {
            if (i > 100) {
                WARNING(
                    "propConst iterates over 100 rounds. Maybe there is a bug");
            }
            break;
        }
        op = ReplaceUses(replaceLoad)(op);
    }

    return op;
}

} // namespace ir

