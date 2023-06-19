#include <mutex>

#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <container_utils.h>
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

std::vector<std::pair<AST, std::pair<Stmt, ReplaceInfo>>>
topoSort(const std::unordered_map<AST, std::pair<Stmt, ReplaceInfo>> &r2w,
         const std::unordered_map<AST, Stmt> &stmts) {
    // DFS post order of a reversed DAG is the original DAG's topogical order
    // We need to find a topogical order of a write-to-read graph, which is the
    // DFS post order of the reversed write-to-read graph, or the DFS post order
    // of the read-to-write graph
    std::vector<std::pair<AST, std::pair<Stmt, ReplaceInfo>>> topo;
    std::unordered_set<AST> visited;
    std::function<void(const AST &x)> recur = [&](const AST &r) {
        if (visited.count(r)) {
            return;
        }
        visited.insert(r);
        if (r2w.count(r)) {
            auto &&w = r2w.at(r).first;
            for (auto &&[nextR, nextRStmt] : stmts) {
                if (nextRStmt == w) {
                    recur(nextR);
                }
            }
            topo.emplace_back(r, r2w.at(r));
        }
    };
    for (auto &&[r, w] : r2w) {
        recur(r);
    }
    return topo;
}

} // Anonymous namespace

Stmt propOneTimeUse(const Stmt &_op, const ID &subAST) {
    auto op = makeReduction(_op);

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call ReplaceUses to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    op = hoistVarOverStmtSeq(op);

    std::unordered_map<AST, std::vector<std::pair<Stmt, ReplaceInfo>>>
        r2wCandidates;
    std::unordered_map<AST, std::vector<Stmt>> r2wMay;
    std::unordered_map<Stmt, std::vector<AST>> w2r;
    std::unordered_map<Stmt,
                       std::vector<std::pair<AST, std::string /* writeIter */>>>
        w2rMay;
    std::unordered_map<AST, Stmt> stmts;
    std::mutex lock;

    // Find dependence A->B that always happen for B which may propagate
    auto finder = FindDeps()
                      .mode(FindDepsMode::KillLater)
                      .type(DEP_RAW)
                      .filterAccess([&](const auto &acc) {
                          return !isOutputting(acc.def_->buffer_->atype());
                      })
                      .filterEarlier([&](const auto &earlier) {
                          return earlier.op_->nodeType() == ASTNodeType::Store;
                      })
                      .filterLater([&](const auto &later) {
                          // pass/remove_write will deal with it (TODO: Really?
                          // What if we want to do interleaved prop_one_time_use
                          // and remove_write?)
                          return later.op_->nodeType() != ASTNodeType::ReduceTo;
                      });
    if (subAST.isValid()) {
        finder = finder.filterSubAST(subAST);
    }
    finder(op, unsyncFunc([&](const Dependence &d) {
               if (d.later2EarlierIter_.isBijective()) {
                   // Check before converting into PBFunc. In prop_one_time_use,
                   // we not only need `singleValued`, but also `bijective`, to
                   // ensure it is really used "one time"
                   if (std::string str = pbFuncSerializedWithTimeout(
                           [](const PBMap &map) { return PBFunc(map); }, 10,
                           d.later2EarlierIter_);
                       !str.empty()) {
                       std::lock_guard _(lock);
                       r2wCandidates[d.later()].emplace_back(
                           // Find dependence A->B that always happen for B
                           // which may propagate
                           d.earlier().as<StmtNode>(),
                           ReplaceInfo{d.earlier_.iter_, d.later_.iter_, str});
                       w2r[d.earlier().as<StmtNode>()].emplace_back(d.later());
                       stmts[d.later()] = d.later_.stmt_;
                   }
               }
           }));

    // Find other potential dependence that may prevent propagation
    //
    // No filter sub-AST because there may be A->C->B dependence for A,B,C in
    // program order. Besides, we need to ensure one-time use in the whole
    // program
    FindDeps()
        .type(DEP_RAW)
        .noProjectOutPrivateAxis(true)
        .filter([&](const AccessPoint &later, const AccessPoint &earlier) {
            return r2wCandidates.count(later.op_) ||
                   w2r.count(earlier.op_.as<StmtNode>());
        })
        .ignoreReductionWAW(false)(op, [&](const Dependence &d) {
            PBSet writeIter = range(d.later2EarlierIter_);
            writeIter = projectOutDims( // Trim paddings
                writeIter, d.earlier_.iter_.size(),
                writeIter.nDims() - d.earlier_.iter_.size());
            r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
            w2rMay[d.earlier().as<StmtNode>()].emplace_back(
                d.later(), toString(writeIter));
        });

    // Filter single-valued and one-time-used
    std::unordered_map<AST, std::pair<Stmt, ReplaceInfo>> r2w;
    for (auto &&[read, writes] : r2wCandidates) {
        // Check single-valued: There should be only 1 possible write statment
        if (writes.size() > 1) {
            continue; // Not single-valued
        }
        ASSERT(writes.size() == 1);
        auto &&write = writes.front();
        if (!r2wMay.count(read) || r2wMay.at(read).size() > 1 ||
            r2wMay.at(read)[0] != write.first) {
            continue; // Not single-valued
        }

        // Check one-time use: All read statements should read different items
        // written by the write statement
        ASSERT(w2rMay.count(write.first));
        PBCtx ctx;
        PBSet writeIterUnion;
        for (auto &&[read, writeIterStr] : w2rMay.at(write.first)) {
            PBSet writeIter = PBSet(ctx, writeIterStr);
            if (writeIterUnion.isValid()) {
                if (!intersect(writeIterUnion, writeIter).empty()) {
                    goto failure; // Not one-time-used
                }
                writeIterUnion = uni(writeIterUnion, writeIter);
            } else {
                writeIterUnion = writeIter;
            }
        }

        r2w[read] = writes.front();

    failure:;
    }

    // To deal with chained propagation, e.g.
    // a = x + 1  // (1)
    // b = a + 1  // (2)
    // c = b + 1  // (3)
    // I. We first apply a topological sort to eusure we handle (1)->(2) before
    // (2)->(3).
    // II. When we are handling (2)->(3), there is already replace[b] = a + 1,
    // so we apply replace on toProp
    auto r2wTopo = topoSort(r2w, stmts);

    std::unordered_map<AST, Expr> replace;
    for (auto &&[read, write] : r2wTopo) {
        ASSERT(write.first->nodeType() == ASTNodeType::Store);
        auto &&toProp =
            ReplaceUses(replace)(write.first.as<StoreNode>()->expr_);
        auto &&repInfo = write.second;

        if (!allIters(toProp).empty()) {
            try {
                auto &&[args, values, cond] =
                    parseSimplePBFunc(repInfo.funcStr_); // later -> earlier
                ASSERT(repInfo.earlierIters_.size() <=
                       values.size()); // maybe padded
                ASSERT(repInfo.laterIters_.size() <= args.size());
                std::unordered_map<std::string, Expr> islVarToNewIter,
                    oldIterToNewIter;
                for (auto &&[newIter, arg] :
                     views::zip(repInfo.laterIters_, args)) {
                    islVarToNewIter[arg] =
                        !newIter.negStep_
                            ? newIter.iter_
                            : makeMul(makeIntConst(-1), newIter.iter_);
                }
                for (auto &&[oldIter, value] :
                     views::zip(repInfo.earlierIters_, values)) {
                    if (oldIter.iter_->nodeType() == ASTNodeType::Var) {
                        oldIterToNewIter[oldIter.iter_.as<VarNode>()->name_] =
                            !oldIter.negStep_
                                ? ReplaceIter(islVarToNewIter)(value)
                                : makeMul(makeIntConst(-1),
                                          ReplaceIter(islVarToNewIter)(value));
                    }
                }
                auto newExpr = ReplaceIter(oldIterToNewIter)(toProp);
                if (!checkNotModified(
                        op, toProp, newExpr, CheckNotModifiedSide::Before,
                        write.first->id(), CheckNotModifiedSide::Before,
                        stmts.at(read)->id())) {
                    goto fail;
                }
                replace[read] = std::move(newExpr);
            } catch (const ParserError &e) {
                // do nothing
            }
        fail:;
        } else {
            if (checkNotModified(
                    op, toProp, CheckNotModifiedSide::Before, write.first->id(),
                    CheckNotModifiedSide::Before, stmts.at(read)->id())) {
                replace[read] = toProp;
            }
        }
    }

    op = ReplaceUses(replace)(op);
    return sinkVar(op);
}

} // namespace freetensor
