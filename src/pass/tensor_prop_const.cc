#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <pass/replace_uses.h>
#include <pass/scalar_prop_const.h>
#include <pass/simplify.h>
#include <pass/tensor_prop_const.h>

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

Stmt tensorPropConst(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = simplifyPass(op);
        op = scalarPropConst(op);

        // Please note that the "reads" might also be reductions.
        // pass/remove_writes cannot replace pass/tensor_prop_const when
        // propagating to ReduceTo nodes. E.g.:
        //
        // ```
        // a = 1  // (1) if (...) {
        //   a += 1  // (2)
        // }
        // ```
        //
        // pass/tensor_prop_const will propagate this case. However
        // pass/remove_writes can not, because statement (1) cannot be removed
        std::unordered_map<AST, std::vector<Stmt>> r2w, r2wMay;
        auto filterMust = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
            if (later.buffer_->tensor().isScalar()) {
                return false;
            }
            if (earlier.op_->nodeType() != ASTNodeType::Store) {
                return false;
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
            ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
            auto &&store = item.second.front().as<StoreNode>();
            replace[item.first] = store->expr_;
        }

        if (replace.empty() || i > 100) {
            if (i > 100) {
                WARNING(
                    "propConst iterates over 100 rounds. Maybe there is a bug");
            }
            break;
        }
        op = ReplaceUses(replace)(op);
    }

    return op;
}

} // namespace ir
