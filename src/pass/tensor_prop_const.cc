#include <itertools.hpp>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <math/parse_pb_expr.h>
#include <pass/replace_iter.h>
#include <pass/replace_uses.h>
#include <pass/scalar_prop_const.h>
#include <pass/tensor_prop_const.h>

namespace freetensor {

namespace {

struct ReplaceInfo {
    std::vector<IterAxis> earlierIters_, laterIters_;
    std::string funcStr_;
};

} // namespace

Stmt tensorPropConst(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
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
        std::unordered_map<AST, std::vector<std::pair<Stmt, ReplaceInfo>>> r2w;
        std::unordered_map<AST, std::vector<Stmt>> r2wMay;
        auto foundMust = [&](const Dependency &d) {
            auto &&expr = d.earlier().as<StoreNode>()->expr_;
            auto &&iters = allIters(expr);
            auto common = lcaStmt(d.later_.stmt_, d.earlier_.stmt_);
            auto dep = d.later2EarlierIter_;
            for (auto &&iter : iters) {
                for (auto c = common; c.isValid(); c = c->parentStmt()) {
                    if (c->nodeType() == ASTNodeType::For) {
                        if (auto &&f = c.as<ForNode>(); f->iter_ == iter) {
                            dep =
                                d.extraCheck(dep, f->id(), DepDirection::Same);
                            if (dep != d.later2EarlierIter_) {
                                // Iterating variable in different iterations
                                return;
                            }
                            break;
                        }
                    }
                }
            }
            if (d.later2EarlierIter_
                    .isSingleValued()) { // Check before converting into PBFunc
                r2w[d.later()].emplace_back(
                    d.earlier().as<StmtNode>(),
                    ReplaceInfo{d.earlier_.iter_, d.later_.iter_,
                                toString(PBFunc(d.later2EarlierIter_))});
            }
        };
        auto foundMay = [&](const Dependency &d) {
            r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
        };
        FindDeps()
            .mode(FindDepsMode::KillLater)
            .type(DEP_RAW)
            .filterAccess([&](const AccessPoint &acc) {
                return !acc.buffer_->tensor()->isScalar();
            })
            .filterEarlier([&](const AccessPoint &earlier) {
                if (earlier.op_->nodeType() != ASTNodeType::Store) {
                    return false;
                }
                auto &&expr = earlier.op_.as<StoreNode>()->expr_;
                if (!allReads(expr).empty()) {
                    // Expressions should contain only constants and iterating
                    // vars
                    return false;
                }
                return true;
            })
            .noProjectOutProvateAxis(true)(op, foundMust);
        FindDeps()
            .type(DEP_RAW)
            .filterLater(
                [&](const AccessPoint &later) { return r2w.count(later.op_); })
            .ignoreReductionWAW(false)(op, foundMay);

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
            ASSERT(item.second.front().first->nodeType() == ASTNodeType::Store);
            auto &&store = item.second.front().first.as<StoreNode>();
            auto &&repInfo = item.second.front().second;

            if (!allIters(store->expr_).empty()) {
                try {
                    auto &&[args, values, cond] =
                        parseSimplePBFunc(repInfo.funcStr_); // later -> earlier
                    ASSERT(repInfo.earlierIters_.size() <=
                           values.size()); // maybe padded
                    ASSERT(repInfo.laterIters_.size() <= args.size());
                    std::unordered_map<std::string, Expr> islVarToNewIter,
                        oldIterToNewIter;
                    for (auto &&[newIter, arg] :
                         iter::zip(repInfo.laterIters_, args)) {
                        islVarToNewIter[arg] =
                            newIter.realIter_ == newIter.iter_
                                ? newIter.iter_
                                : makeMul(makeIntConst(-1), newIter.realIter_);
                    }
                    for (auto &&[oldIter, value] :
                         iter::zip(repInfo.earlierIters_, values)) {
                        if (oldIter.realIter_->nodeType() == ASTNodeType::Var) {
                            oldIterToNewIter[oldIter.realIter_.as<VarNode>()
                                                 ->name_] =
                                oldIter.realIter_ == oldIter.iter_
                                    ? ReplaceIter(islVarToNewIter)(value)
                                    : makeMul(
                                          makeIntConst(-1),
                                          ReplaceIter(islVarToNewIter)(value));
                        }
                    }
                    replace[item.first] =
                        ReplaceIter(oldIterToNewIter)(store->expr_);
                } catch (const ParserError &e) {
                    // do nothing
                }
            } else {
                replace[item.first] = store->expr_;
            }
        }

        if (replace.empty() || i > 100) {
            if (i > 100) {
                WARNING("tensor_prop_const iterates over 100 rounds. Maybe "
                        "there is a bug");
            }
            break;
        }
        op = ReplaceUses(replace)(op);
    }

    return op;
}

} // namespace freetensor
