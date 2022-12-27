#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <container_utils.h>
#include <math/parse_pb_expr.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/remove_writes.h>
#include <pass/replace_iter.h>
#include <pass/sink_var.h>

namespace freetensor {

namespace {

struct ReplaceInfo {
    std::vector<IterAxis> earlierIters_, laterIters_;
    std::string funcStr_;
};

} // Anonymous namespace

static bool sameParent(const Stmt &x, const Stmt &y) {
    return x->parentCtrlFlow() == y->parentCtrlFlow();
}

static ReduceOp combineReduce(ReduceOp reduceOp) {
    // x -= 1; x -= 2; ==> x -= 1 + 2
    switch (reduceOp) {
    case ReduceOp::Add:
    case ReduceOp::Sub: // !!!
        return ReduceOp::Add;
    case ReduceOp::Mul:
        return ReduceOp::Mul;
    case ReduceOp::Min:
        return ReduceOp::Min;
    case ReduceOp::Max:
        return ReduceOp::Max;
    case ReduceOp::LAnd:
        return ReduceOp::LAnd;
    case ReduceOp::LOr:
        return ReduceOp::LOr;
    default:
        ASSERT(false);
    }
}

static Expr makeReduce(ReduceOp reduceOp, const Expr &lhs, const Expr &rhs) {
    switch (reduceOp) {
    case ReduceOp::Add:
        return makeAdd(lhs, rhs);
    case ReduceOp::Sub:
        return makeSub(lhs, rhs);
    case ReduceOp::Mul:
        return makeMul(lhs, rhs);
    case ReduceOp::Max:
        return makeMax(lhs, rhs);
    case ReduceOp::Min:
        return makeMin(lhs, rhs);
    case ReduceOp::LAnd:
        return makeLAnd(lhs, rhs);
    case ReduceOp::LOr:
        return makeLOr(lhs, rhs);
    default:
        ASSERT(false);
    }
}

static std::vector<std::tuple<Stmt, Stmt, PBSet, ReplaceInfo>> topoSort(
    const std::vector<std::tuple<Stmt, Stmt, PBSet, ReplaceInfo>> &overwrites) {
    // DFS post order of a reversed DAG is the original DAG's topogical order
    // We need to find a topogical order of a earlier-to-later graph, which is
    // the DFS post order of the reversed earlier-to-later graph, or the DFS
    // post order of the later-to-earlier graph
    std::vector<std::tuple<Stmt, Stmt, PBSet, ReplaceInfo>> topo;
    std::unordered_set<AST> visited;
    std::function<void(const Stmt &x)> recur = [&](const Stmt &later) {
        if (visited.count(later)) {
            return;
        }
        visited.insert(later);
        for (auto &&[_later, _earlier, _1, _2] : overwrites) {
            if (_later == later) {
                recur(_earlier);
                topo.emplace_back(_later, _earlier, _1, _2);
            }
        }
    };
    for (auto &&[later, earlier, _1, _2] : overwrites) {
        recur(later);
    }
    return topo;
}

void FindLoopInvariantWrites::visit(const For &op) {
    loopStack_.emplace_back(op);
    BaseClass::visit(op);
    loopStack_.pop_back();
}

void FindLoopInvariantWrites::visit(const If &op) {
    ifStack_.emplace_back(op);
    BaseClass::visit(op);
    ifStack_.pop_back();
}

void FindLoopInvariantWrites::visit(const VarDef &op) {
    defDepth_[op->name_] = loopStack_.size();
    BaseClass::visit(op);
    defDepth_.erase(op->name_);
}

void FindLoopInvariantWrites::visit(const Store &op) {
    BaseClass::visit(op);
    if (singleDefId_.isValid() && def(op->var_)->id() != singleDefId_) {
        return;
    }
    Expr cond;
    For innerMostLoop;

    for (int i = (int)(loopStack_.size()) - 1, iEnd = defDepth_.at(op->var_);
         i >= iEnd; i--) {
        auto &&item = loopStack_[i];
        if (item->property_->parallel_ != serialScope) {
            continue;
        }
        auto rbegin =
            makeAdd(item->begin_,
                    makeMul(makeSub(item->len_, makeIntConst(1)), item->step_));
        Expr thisCond;
        for (auto &&idx : op->indices_) {
            if (isVariant(variantExpr_, {idx, op}, item->id())) {
                goto fail;
            }
        }
        for (auto &&branch : ifStack_) {
            if (isVariant(variantExpr_, {branch->cond_, branch}, item->id())) {
                goto fail;
            }
        }
        thisCond = makeEQ(makeVar(item->iter_), rbegin);
        if (!cond.isValid()) {
            innerMostLoop = item;
            cond = thisCond;
        } else {
            cond = makeLAnd(cond, thisCond);
        }
        continue;
    fail:;
    }

    if (cond.isValid()) {
        results_[op] = std::make_tuple(def(op->var_), cond, innerMostLoop);
    }
}

Stmt RemoveWrites::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    std::vector<Stmt> stmts;
    for (auto &&stmt : op->stmts_) {
        if (stmt->nodeType() != ASTNodeType::StmtSeq ||
            !stmt.as<StmtSeqNode>()->stmts_.empty()) {
            stmts.emplace_back(stmt);
        }
    }
    op->stmts_ = std::move(stmts);
    return op;
}

Stmt RemoveWrites::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->body_->nodeType() == ASTNodeType::StmtSeq &&
        op->body_.as<StmtSeqNode>()->stmts_.empty()) {
        return makeStmtSeq({});
    }
    return op;
}

Stmt RemoveWrites::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    auto thenValid = op->thenCase_->nodeType() != ASTNodeType::StmtSeq ||
                     !op->thenCase_.as<StmtSeqNode>()->stmts_.empty();
    auto elseValid = op->elseCase_.isValid() &&
                     (op->elseCase_->nodeType() != ASTNodeType::StmtSeq ||
                      !op->elseCase_.as<StmtSeqNode>()->stmts_.empty());
    if (!thenValid && !elseValid) {
        return makeStmtSeq({});
    }
    if (!elseValid) {
        return makeIf(op->cond_, op->thenCase_, op->metadata(), op->id());
    }
    if (!thenValid) {
        return makeIf(makeLNot(op->cond_), op->elseCase_, op->metadata(),
                      op->id());
    }
    return op;
}

Stmt removeWrites(const Stmt &_op, const ID &singleDefId) {
    auto op = makeReduction(_op);

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call RemoveWrites to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    op = hoistVarOverStmtSeq(op);

    auto variantExpr = findLoopVariance(op);
    FindLoopInvariantWrites type2Finder(variantExpr.first, singleDefId);
    type2Finder(op);
    auto type2Results = type2Finder.results();

    std::unordered_set<VarDef> suspect;
    for (auto &&[store, item] : type2Results) {
        auto &&[def, cond, loop] = item;
        suspect.insert(def);
    }

    // Used to prune
    std::unordered_set<Stmt> selfDependentReduces;
    auto foundSelfDependent = [&](const Dependence &d) {
        selfDependentReduces.insert(d.later().as<StmtNode>());
    };
    FindDeps()
        .type(DEP_WAW)
        .filterLater([&](const AccessPoint &later) {
            return later.op_->nodeType() == ASTNodeType::ReduceTo;
        })
        .filter([&](const AccessPoint &later, const AccessPoint &earlier) {
            return earlier.op_ == later.op_;
        })
        .ignoreReductionWAW(false)(op, foundSelfDependent);

    PBCtx presburger;
    // {(later, earlier, toKill, replaceInfo)}
    std::vector<std::tuple<Stmt, Stmt, PBSet, ReplaceInfo>> overwrites;
    std::unordered_map<Stmt, std::unordered_set<AST>> usesRAW; // W -> R
    std::unordered_map<Stmt, std::unordered_set<AST>> usesWAR; // W -> R
    std::unordered_map<Stmt, PBSet> kill;
    auto foundOverwriteStore = [&](const Dependence &d) {
        auto earlier = d.earlier().as<StmtNode>();
        auto later = d.later().as<StmtNode>();
        if (!kill.count(earlier)) {
            kill[earlier] =
                PBSet(presburger, toString(domain(d.earlierIter2Idx_)));
        }
        overwrites.emplace_back(
            later, earlier,
            PBSet(presburger, toString(range(d.later2EarlierIter_))),
            ReplaceInfo{});
        suspect.insert(d.def());
    };
    auto foundOverwriteReduce = [&](const Dependence &d) {
        if (d.later() != d.earlier() &&
            (!selfDependentReduces.count(d.later().as<StmtNode>()) ||
             sameParent(d.later_.stmt_, d.earlier_.stmt_))) {
            if (d.later2EarlierIter_.isSingleValued()) {
                auto earlier = d.earlier().as<StmtNode>();
                auto later = d.later().as<StmtNode>();
                if (!kill.count(earlier)) {
                    kill[earlier] =
                        PBSet(presburger, toString(domain(d.earlierIter2Idx_)));
                }
                overwrites.emplace_back(
                    later, earlier,
                    PBSet(presburger, toString(range(d.later2EarlierIter_))),
                    ReplaceInfo{d.earlier_.iter_, d.later_.iter_,
                                toString(PBFunc(d.later2EarlierIter_))});
                suspect.insert(d.def());
            }
        }
    };
    auto foundUse = [&](const Dependence &d) {
        if (d.later()->nodeType() != ASTNodeType::Store &&
            d.earlier()->nodeType() != ASTNodeType::Load &&
            d.earlier() != d.later()) {
            usesRAW[d.earlier().as<StmtNode>()].emplace(d.later());
        }
        if (d.earlier()->nodeType() != ASTNodeType::Store &&
            d.later()->nodeType() != ASTNodeType::Load &&
            d.earlier() != d.later()) {
            usesWAR[d.later().as<StmtNode>()].emplace(d.earlier());
        }

        if (d.later()->nodeType() != ASTNodeType::Store &&
            d.earlier()->nodeType() == ASTNodeType::Store &&
            d.earlier() != d.later()) {
            if (type2Results.count(d.earlier().as<StoreNode>())) {
                auto &&[def, cond, innerMostLoop] =
                    type2Results.at(d.earlier().as<StoreNode>());
                if (lcaStmt(innerMostLoop, d.later_.stmt_) == innerMostLoop) {
                    type2Results.erase(d.earlier().as<StoreNode>());
                }
            }
        }
    };

    FindDeps()
        .type(DEP_WAW)
        .filterAccess([&](const AccessPoint &acc) {
            return !singleDefId.isValid() || acc.def_->id() == singleDefId;
        })
        .filterLater([&](const AccessPoint &later) {
            return later.op_->nodeType() == ASTNodeType::Store;
        })
        .ignoreReductionWAW(false)
        .noProjectOutPrivateAxis(true)(op, foundOverwriteStore);
    FindDeps()
        .mode(FindDepsMode::KillLater)
        .type(DEP_WAW)
        .filterAccess([&](const AccessPoint &acc) {
            return !singleDefId.isValid() || acc.def_->id() == singleDefId;
        })
        .filterLater([&](const AccessPoint &later) {
            return later.op_->nodeType() == ASTNodeType::ReduceTo;
        })
        .ignoreReductionWAW(false)
        .noProjectOutPrivateAxis(true)(op, foundOverwriteReduce);
    FindDeps()
        .filterAccess([&](const AccessPoint &access) {
            return suspect.count(access.def_);
        })
        .ignoreReductionWAW(false)(op, foundUse);

    std::unordered_set<Stmt> redundant;
    std::unordered_map<Stmt, Stmt> replacement;

    // Type 1

    // Make sure there is no read between the overwriting two writes
    for (auto i = overwrites.begin(); i != overwrites.end();) {
        auto &&[later, earlier, _1, _2] = *i;
        if (usesRAW.count(earlier) && usesWAR.count(later) &&
            hasIntersect(usesRAW.at(earlier), usesWAR.at(later))) {
            i = overwrites.erase(i);
        } else {
            i++;
        }
    }

    // Make sure all instance of earlier is overwritten by later. E.g.,
    //
    // for i
    //   if i < 5
    //     a = x
    //   a = y
    //
    // is OK, while
    //
    // for i
    //   a = x
    //   if i < 5
    //     a = y
    //
    // is not.
    //
    // We also support a more complex case, where multiple laters overwirtes a
    // single earlier. E.g.,
    //
    // for i
    //   a = x
    //   if i < 5
    //     a = y
    //   else
    //     a = z
    for (auto &&[later, earlier, thisKill, _] : overwrites) {
        kill[earlier] = subtract(std::move(kill.at(earlier)), thisKill);
    }
    for (auto i = overwrites.begin(); i != overwrites.end();) {
        auto &&[later, earlier, _1, _2] = *i;
        if (later == earlier) {
            i = overwrites.erase(i);
        } else if (!kill.at(earlier).empty()) {
            i = overwrites.erase(i);
        } else {
            i++;
        }
    }

    // To deal with chained propagation, e.g.
    //
    // a += x  // (1)
    // a += y  // (2)
    // a += z  // (3)
    //
    // I. We first apply a topological sort to eusure we handle (1)->(2) before
    // (2)->(3).
    // II. When we are handling (2)->(3), there is already replacement[(2)] =
    // xxxx. We start from that replacement and keep replacing
    overwrites = topoSort(overwrites);

    for (auto &&[_later, _earlier, _, repInfo] : overwrites) {
        ASSERT(!replacement.count(_later));
        auto &&later = _later;
        auto &&earlier =
            replacement.count(_earlier) ? replacement.at(_earlier) : _earlier;

        if (later->nodeType() == ASTNodeType::Store) {
            redundant.insert(_earlier);
        } else {
            ASSERT(later->nodeType() == ASTNodeType::ReduceTo);

            Expr expr = earlier->nodeType() == ASTNodeType::Store
                            ? earlier.as<StoreNode>()->expr_
                            : earlier.as<ReduceToNode>()->expr_;

            if (!allIters(expr).empty()) {
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
                            oldIterToNewIter[oldIter.iter_.as<VarNode>()
                                                 ->name_] =
                                !oldIter.negStep_
                                    ? ReplaceIter(islVarToNewIter)(value)
                                    : makeMul(
                                          makeIntConst(-1),
                                          ReplaceIter(islVarToNewIter)(value));
                        }
                    }
                    auto newExpr = ReplaceIter(oldIterToNewIter)(expr);
                    if (!checkNotModified(
                            op, expr, newExpr, CheckNotModifiedSide::After,
                            earlier->id(), CheckNotModifiedSide::Before,
                            later->id())) {
                        continue;
                    }
                    auto l = later.as<ReduceToNode>();
                    if (earlier->nodeType() == ASTNodeType::Store) {
                        redundant.insert(_earlier);
                        replacement[_later] =
                            makeStore(l->var_, l->indices_,
                                      makeReduce(l->op_, newExpr, l->expr_),
                                      later->metadata(), later->id());
                    } else if (earlier.as<ReduceToNode>()->op_ == l->op_) {
                        redundant.insert(_earlier);
                        replacement[_later] =
                            makeReduceTo(l->var_, l->indices_, l->op_,
                                         makeReduce(combineReduce(l->op_),
                                                    newExpr, l->expr_),
                                         false, later->metadata(), later->id());
                    }
                } catch (const ParserError &e) {
                    // do nothing
                }
            } else {
                if (checkNotModified(
                        op, expr, CheckNotModifiedSide::After, earlier->id(),
                        CheckNotModifiedSide::Before, later->id())) {
                    auto l = later.as<ReduceToNode>();
                    if (earlier->nodeType() == ASTNodeType::Store) {
                        redundant.insert(_earlier);
                        replacement[_later] =
                            makeStore(l->var_, l->indices_,
                                      makeReduce(l->op_, expr, l->expr_),
                                      later->metadata(), later->id());
                    } else if (earlier.as<ReduceToNode>()->op_ == l->op_) {
                        redundant.insert(_earlier);
                        replacement[_later] = makeReduceTo(
                            l->var_, l->indices_, l->op_,
                            makeReduce(combineReduce(l->op_), expr, l->expr_),
                            false, later->metadata(), later->id());
                    }
                }
            }
        }
    }

    // Type 2
    for (auto &&[_store, item] : type2Results) {
        auto &&[def, cond, loop] = item;
        auto store = _store.as<StmtNode>();
        replacement.emplace(store, makeIf(cond, store));
    }

    op = RemoveWrites(redundant, replacement)(op);
    return sinkVar(op);
}

} // namespace freetensor
