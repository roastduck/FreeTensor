#include <functional>
#include <set>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/remove_writes.h>
#include <pass/sink_var.h>

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
    Visitor::visit(op);
    loopStack_.pop_back();
}

void FindLoopInvariantWrites::visit(const If &op) {
    ifStack_.emplace_back(op);
    Visitor::visit(op);
    ifStack_.pop_back();
}

void FindLoopInvariantWrites::visit(const VarDef &op) {
    defDepth_[op->name_] = loopStack_.size();
    defs_[op->name_] = op;
    Visitor::visit(op);
    defs_.erase(op->name_);
    defDepth_.erase(op->name_);
}

void FindLoopInvariantWrites::visit(const Store &op) {
    Visitor::visit(op);
    Expr cond;
    for (int i = (int)(loopStack_.size()) - 1, iEnd = defDepth_.at(op->var_);
         i >= iEnd; i--) {
        auto &&item = loopStack_[i];
        if (!item->property_.parallel_.empty()) {
            continue;
        }
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
        thisCond =
            makeEQ(makeVar(item->iter_), makeSub(item->end_, makeIntConst(1)));
        cond = cond.isValid() ? makeLAnd(cond, thisCond) : thisCond;
        continue;
    fail:;
    }

    if (cond.isValid()) {
        results_.emplace_back(defs_.at(op->var_), op, cond);
    }
}

Stmt removeWrites(const Stmt &_op) {
    auto op = makeReduction(_op);

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call RemoveWrites to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    op = hoistVarOverStmtSeq(op);

    auto variantExpr = findLoopVariance(op);
    FindLoopInvariantWrites type2Finder(variantExpr.first);
    type2Finder(op);

    std::unordered_set<VarDef> suspect;
    for (auto &&[def, store, cond] : type2Finder.results()) {
        suspect.insert(def);
    }

    // {(later, earlier)}
    std::set<std::pair<Stmt, Stmt>> overwrites;
    std::set<std::pair<AST, Stmt>> usesRAW;
    std::set<std::pair<Stmt, AST>> usesWAR;
    auto filterOverwrite = [&](const AccessPoint &later,
                               const AccessPoint &earlier) {
        if (later.op_.get() == earlier.op_.get()) {
            return false;
        }
        return later.op_->nodeType() == ASTNodeType::Store ||
               sameParent(later.cursor_, earlier.cursor_);
    };
    auto foundOverwrite = [&](const Dependency &d) {
        overwrites.emplace(d.later().as<StmtNode>(),
                           d.earlier().as<StmtNode>());
        suspect.insert(d.def());
    };
    auto filterUse = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return suspect.count(later.def_);
    };
    auto foundUse = [&](const Dependency &d) {
        if (d.later()->nodeType() != ASTNodeType::Store &&
            d.earlier()->nodeType() != ASTNodeType::Load) {
            usesRAW.emplace(d.later(), d.earlier().as<StmtNode>());
        }
        if (d.earlier()->nodeType() != ASTNodeType::Store &&
            d.later()->nodeType() != ASTNodeType::Load) {
            usesWAR.emplace(d.later().as<StmtNode>(), d.earlier());
        }
    };

    findDeps(op, {{}}, foundOverwrite, FindDepsMode::KillEarlier, DEP_WAW,
             filterOverwrite, false);
    findDeps(op, {{}}, foundUse, FindDepsMode::Dep, DEP_ALL, filterUse, false);

    std::unordered_set<Stmt> redundant;
    std::unordered_map<Stmt, Stmt> replacement;

    // Type 1
    std::set<std::pair<Stmt, Stmt>> visited;
    std::function<void(const std::pair<Stmt, Stmt> &)> visitType1 =
        [&](const std::pair<Stmt, Stmt> &item) {
            if (visited.count(item)) {
                return;
            }
            visited.insert(item);
            auto &&_later = item.first, &&_earlier = item.second;

            for (auto &&use : overwrites) {
                if (use.first == _earlier && !redundant.count(use.second)) {
                    // In the case of:
                    // (1) A = X
                    // (2) A += Y
                    // (3) A += Z
                    // if we handle (2)-(3) first, which resulting to `A += Y +
                    // Z`, we cannot handle (1) then. So, we handle (1)-(2)
                    // before (2)-(3)
                    visitType1(use);
                }
            }

            for (auto &&use : usesRAW) {
                if (use.second == _earlier &&
                    (use.first->nodeType() == ASTNodeType::Load ||
                     !redundant.count(use.first.as<StmtNode>())) &&
                    usesWAR.count(std::make_pair(_later, use.first))) {
                    return;
                }
            }

            if (redundant.count(_later) || redundant.count(_earlier)) {
                return;
            }
            auto later =
                replacement.count(_later) ? replacement.at(_later) : _later;
            auto earlier = replacement.count(_earlier)
                               ? replacement.at(_earlier)
                               : _earlier;

            if (later->nodeType() == ASTNodeType::Store) {
                redundant.insert(_earlier);
            } else {
                ASSERT(later->nodeType() == ASTNodeType::ReduceTo);

                Expr expr = earlier->nodeType() == ASTNodeType::Store
                                ? earlier.as<StoreNode>()->expr_
                                : earlier.as<ReduceToNode>()->expr_;

                if (!checkNotModified(
                        op, expr, CheckNotModifiedSide::After, earlier->id(),
                        CheckNotModifiedSide::Before, later->id())) {
                    return;
                }

                auto l = later.as<ReduceToNode>();
                if (earlier->nodeType() == ASTNodeType::Store) {
                    redundant.insert(_earlier);
                    replacement[_later] = makeStore(
                        later->id(), l->var_, l->indices_,
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
        };
    for (auto &&item : overwrites) {
        visitType1(item);
    }

    // Type 2
    for (auto &&[def, _store, cond] : type2Finder.results()) {
        auto store = _store.as<StmtNode>();
        for (auto &&use : usesRAW) {
            if (use.second == store &&
                usesWAR.count(std::make_pair(store, use.first))) {
                goto type2Fail;
            }
        }
        replacement.emplace(store, makeIf("", cond, store));
        continue;
    type2Fail:;
    }

    op = RemoveWrites(redundant, replacement)(op);
    return sinkVar(op);
}

} // namespace ir

