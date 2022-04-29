#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <container_utils.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/remove_writes.h>
#include <pass/sink_var.h>

namespace freetensor {

static bool sameParent(const Stmt &x, const Stmt &y) {
    return x->parentCtrlFlow() == y->parentCtrlFlow();
}

static Expr makeReduce(ReduceOp reduceOp, const Expr &lhs, const Expr &rhs) {
    switch (reduceOp) {
    case ReduceOp::Add:
        return makeAdd(lhs, rhs);
    case ReduceOp::Mul:
        return makeMul(lhs, rhs);
    case ReduceOp::Max:
        return makeMax(lhs, rhs);
    case ReduceOp::Min:
        return makeMin(lhs, rhs);
    default:
        ASSERT(false);
    }
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
            if (isVariant(variantExpr_, idx, item->id())) {
                goto fail;
            }
        }
        for (auto &&branch : ifStack_) {
            if (isVariant(variantExpr_, branch->cond_, item->id())) {
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
        return makeStmtSeq("", {});
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
        return makeStmtSeq("", {});
    }
    if (!elseValid) {
        return makeIf(op->id(), op->cond_, op->thenCase_);
    }
    if (!thenValid) {
        return makeIf(op->id(), makeLNot(op->cond_), op->elseCase_);
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
    auto filterSelfDependent = [&](const AccessPoint &later,
                                   const AccessPoint &earlier) {
        return later.op_->nodeType() == ASTNodeType::ReduceTo &&
               earlier.op_ == later.op_;
    };
    auto foundSelfDependent = [&](const Dependency &d) {
        selfDependentReduces.insert(d.later().as<StmtNode>());
    };
    findDeps(op, {{}}, foundSelfDependent, FindDepsMode::Dep, DEP_WAW,
             filterSelfDependent, false);

    PBCtx presburger;
    // {(later, earlier, toKill)}
    std::vector<std::tuple<Stmt, Stmt, PBSet>> overwrites;
    std::unordered_map<Stmt, std::unordered_set<AST>> usesRAW; // W -> R
    std::unordered_map<Stmt, std::unordered_set<AST>> usesWAR; // W -> R
    std::unordered_map<Stmt, PBSet> kill;
    auto filterOverwriteStore = [&](const AccessPoint &later,
                                    const AccessPoint &earlier) {
        if (singleDefId.isValid() && later.def_->id() != singleDefId) {
            return false;
        }
        return later.op_->nodeType() == ASTNodeType::Store;
    };
    auto filterOverwriteReduce = [&](const AccessPoint &later,
                                     const AccessPoint &earlier) {
        if (singleDefId.isValid() && later.def_->id() != singleDefId) {
            return false;
        }
        return later.op_->nodeType() == ASTNodeType::ReduceTo;
    };
    auto foundOverwriteStore = [&](const Dependency &d) {
        auto earlier = d.earlier().as<StmtNode>();
        auto later = d.later().as<StmtNode>();
        if (!kill.count(earlier)) {
            kill[earlier] = PBSet(presburger, toString(domain(d.omap_)));
        }
        overwrites.emplace_back(later, earlier,
                                PBSet(presburger, toString(range(d.dep_))));
        suspect.insert(d.def());
    };
    auto foundOverwriteReduce = [&](const Dependency &d) {
        if (d.later() != d.earlier() &&
            (!selfDependentReduces.count(d.later().as<StmtNode>()) ||
             sameParent(d.later_.stmt_, d.earlier_.stmt_))) {
            if (d.earlier()->nodeType() == ASTNodeType::Store &&
                d.earlier().as<StoreNode>()->expr_->isConst()) {
                goto is_overwrite;
            } else if (d.earlier()->nodeType() == ASTNodeType::ReduceTo &&
                       d.earlier().as<ReduceToNode>()->expr_->isConst()) {
                goto is_overwrite;
            } else if (sameParent(d.later_.stmt_, d.earlier_.stmt_)) {
                goto is_overwrite;
            }
            return;

        is_overwrite:
            auto earlier = d.earlier().as<StmtNode>();
            auto later = d.later().as<StmtNode>();
            if (!kill.count(earlier)) {
                kill[earlier] = PBSet(presburger, toString(domain(d.omap_)));
            }
            overwrites.emplace_back(later, earlier,
                                    PBSet(presburger, toString(range(d.dep_))));
            suspect.insert(d.def());
        }
    };
    auto filterUse = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return suspect.count(later.def_);
    };
    auto foundUse = [&](const Dependency &d) {
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

    findDeps(op, {{}}, foundOverwriteStore, FindDepsMode::Dep, DEP_WAW,
             filterOverwriteStore, false, true, true);
    findDeps(op, {{}}, foundOverwriteReduce, FindDepsMode::KillLater, DEP_WAW,
             filterOverwriteReduce, false, true, true);
    findDeps(op, {{}}, foundUse, FindDepsMode::Dep, DEP_ALL, filterUse, false);

    std::unordered_set<Stmt> redundant;
    std::unordered_map<Stmt, Stmt> replacement;

    // Type 1
    for (auto i = overwrites.begin(); i != overwrites.end();) {
        auto &&[later, earlier, _] = *i;
        if (usesRAW.count(earlier) && usesWAR.count(later) &&
            hasIntersect(usesRAW.at(earlier), usesWAR.at(later))) {
            i = overwrites.erase(i);
        } else {
            i++;
        }
    }
    for (auto &&[later, earlier, thisKill] : overwrites) {
        kill[earlier] = subtract(std::move(kill.at(earlier)), thisKill);
    }
    for (auto i = overwrites.begin(); i != overwrites.end();) {
        auto &&[later, earlier, _] = *i;
        if (!kill.at(earlier).empty()) {
            i = overwrites.erase(i);
        } else {
            i++;
        }
    }
    for (auto i = overwrites.begin(); i != overwrites.end(); i++) {
        auto &&[_later, _earlier, _] = *i;

        if (_later == _earlier) {
            continue;
        }
        if (redundant.count(_later) || redundant.count(_earlier)) {
            continue;
        }
        auto later =
            replacement.count(_later) ? replacement.at(_later) : _later;
        auto earlier =
            replacement.count(_earlier) ? replacement.at(_earlier) : _earlier;

        if (later->nodeType() == ASTNodeType::Store) {
            redundant.insert(_earlier);
        } else {
            ASSERT(later->nodeType() == ASTNodeType::ReduceTo);

            Expr expr = earlier->nodeType() == ASTNodeType::Store
                            ? earlier.as<StoreNode>()->expr_
                            : earlier.as<ReduceToNode>()->expr_;

            if (!checkNotModified(op, expr, CheckNotModifiedSide::After,
                                  earlier->id(), CheckNotModifiedSide::Before,
                                  later->id())) {
                continue;
            }

            auto l = later.as<ReduceToNode>();
            if (earlier->nodeType() == ASTNodeType::Store) {
                redundant.insert(_earlier);
                replacement[_later] =
                    makeStore(later->id(), l->var_, l->indices_,
                              makeReduce(l->op_, earlier.as<StoreNode>()->expr_,
                                         l->expr_));
            } else if (earlier.as<ReduceToNode>()->op_ == l->op_) {
                redundant.insert(_earlier);
                replacement[_later] = makeReduceTo(
                    later->id(), l->var_, l->indices_, l->op_,
                    makeReduce(l->op_, earlier.as<ReduceToNode>()->expr_,
                               l->expr_),
                    false);
            }
        }

        auto j = i;
        for (j++; j != overwrites.end(); j++) {
            auto &[__later, __earlier, _] = *j;
            if (__later == _earlier) {
                __later = _later;
            }
        }
    }

    // Type 2
    for (auto &&[_store, item] : type2Results) {
        auto &&[def, cond, loop] = item;
        auto store = _store.as<StmtNode>();
        replacement.emplace(store, makeIf("", cond, store));
    }

    op = RemoveWrites(redundant, replacement)(op);
    return sinkVar(op);
}

} // namespace freetensor
