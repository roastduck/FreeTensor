#include <itertools.hpp>

#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <math/parse_pb_expr.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/prop_one_time_use.h>
#include <pass/replace_iter.h>
#include <pass/replace_uses.h>
#include <pass/sink_var.h>

namespace freetensor {

namespace {

struct ReplaceInfo {
    std::vector<IterAxis> earlierIters_, laterIters_;
    std::string funcStr_;
};

} // namespace

Stmt propOneTimeUse(const Stmt &_op) {
    auto op = makeReduction(_op);

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call ReplaceUses to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    op = hoistVarOverStmtSeq(op);

    for (int i = 0;; i++) {
        std::unordered_map<AST, std::vector<std::pair<Stmt, ReplaceInfo>>> r2w;
        std::unordered_map<AST, std::vector<Stmt>> r2wMay;
        std::unordered_map<Stmt, std::vector<AST>> w2r, w2rMay;
        std::unordered_map<AST, Stmt> stmts;
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
            return true;
        };
        auto foundMust = [&](const Dependency &d) {
            if (d.later2EarlierIter_.isBijective()) {
                // Check before converting into PBFunc. In prop_one_time_use, we
                // not only need `singleValued`, but also `bijective`, to ensure
                // it is really used "one time"
                r2w[d.later()].emplace_back(
                    d.earlier().as<StmtNode>(),
                    ReplaceInfo{d.earlier_.iter_, d.later_.iter_,
                                toString(PBFunc(d.later2EarlierIter_))});
                w2r[d.earlier().as<StmtNode>()].emplace_back(d.later());
                stmts[d.later()] = d.later_.stmt_;
            }
        };
        auto filterMay = [&](const AccessPoint &later,
                             const AccessPoint &earlier) {
            return r2w.count(later.op_) ||
                   w2r.count(earlier.op_.as<StmtNode>());
        };
        auto foundMay = [&](const Dependency &d) {
            r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
            w2rMay[d.earlier().as<StmtNode>()].emplace_back(d.later());
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
                r2wMay.at(item.first)[0] != item.second.front().first) {
                continue;
            }
            if (!w2rMay.count(item.second.front().first) ||
                w2rMay.at(item.second.front().first).size() > 1 ||
                w2rMay.at(item.second.front().first)[0] != item.first) {
                continue;
            }
            ASSERT(item.second.front().first->nodeType() == ASTNodeType::Store);
            auto &&store = item.second.front().first.as<StoreNode>();
            auto &&repInfo = item.second.front().second;

            if (!allIters(store->expr_).empty()) {
                try {
                    auto &&[args, values, cond] =
                        parsePBFunc(repInfo.funcStr_); // later -> earlier
                    ASSERT(repInfo.earlierIters_.size() <=
                           values.size()); // maybe padded
                    ASSERT(repInfo.laterIters_.size() <= args.size());
                    std::unordered_map<std::string, Expr> islVarToNewIter,
                        oldIterToNewIter;
                    for (auto &&[newIter, arg] :
                         iter::zip(repInfo.laterIters_, args)) {
                        islVarToNewIter[arg] = newIter.iter_;
                    }
                    for (auto &&[oldIter, value] :
                         iter::zip(repInfo.earlierIters_, values)) {
                        if (oldIter.iter_->nodeType() == ASTNodeType::Var) {
                            oldIterToNewIter[oldIter.iter_.as<VarNode>()
                                                 ->name_] =
                                ReplaceIter(islVarToNewIter)(value);
                        }
                    }
                    auto newExpr = ReplaceIter(oldIterToNewIter)(store->expr_);
                    if (!checkNotModified(op, store->expr_, newExpr,
                                          CheckNotModifiedSide::Before,
                                          store->id(),
                                          CheckNotModifiedSide::Before,
                                          stmts.at(item.first)->id())) {
                        goto fail;
                    }
                    replace[item.first] = std::move(newExpr);
                } catch (const ParserError &e) {
                    // do nothing
                }
            fail:;
            } else {
                if (checkNotModified(op, store->expr_,
                                     CheckNotModifiedSide::Before, store->id(),
                                     CheckNotModifiedSide::Before,
                                     stmts.at(item.first)->id())) {
                    replace[item.first] = store->expr_;
                }
            }
        }

        if (replace.empty() || i > 100) {
            if (i > 100) {
                WARNING(
                    "prop_one_time_use iterates over 100 rounds. Maybe there "
                    "is a bug");
            }
            break;
        }
        op = ReplaceUses(replace)(op);
    }

    return sinkVar(op);
}

} // namespace freetensor
