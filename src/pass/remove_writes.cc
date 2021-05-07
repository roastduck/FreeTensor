#include <functional>
#include <set>

#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/remove_writes.h>

namespace ir {

static bool sameParent(const Cursor &x, const Cursor &y) {
    if (!x.hasOuter() && !y.hasOuter()) {
        return true;
    }
    if (x.hasOuter() && y.hasOuter() && x.outer().id() == y.outer().id()) {
        return true;
    }
    return false;
}

static Expr makeReduce(ReduceOp reduceOp, const Expr &lhs, const Expr &rhs) {
    switch (reduceOp) {
    case ReduceOp::Add:
        return makeAdd(lhs, rhs);
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
    Visitor::visit(op);
    defDepth_.erase(op->name_);
}

void FindLoopInvariantWrites::visit(const Store &op) {
    Visitor::visit(op);
    Expr cond;
    for (int i = (int)(loopStack_.size()) - 1, iEnd = defDepth_.at(op->var_);
         i >= iEnd; i--) {
        auto &&item = loopStack_[i];
        if (!item->parallel_.empty()) {
            continue;
        }
        Expr thisCond;
        for (auto &&idx : op->indices_) {
            if (variantExpr_.count(idx) &&
                variantExpr_.at(idx).count(item->id())) {
                goto fail;
            }
        }
        for (auto &&branch : ifStack_) {
            if (variantExpr_.count(branch->cond_) &&
                variantExpr_.at(branch->cond_).count(item->id())) {
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
        results_.emplace_back(op, cond);
    }
}

Stmt removeWrites(const Stmt &_op) {
    auto op = makeReduction(_op);

    // {(later, earlier)}
    std::set<std::pair<Stmt, Stmt>> overwrites;
    std::set<std::pair<Expr, Stmt>> usesRAW;
    std::set<std::pair<Stmt, Expr>> usesWAR;
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
    };
    auto foundUse = [&](const Dependency &d) {
        if (d.later()->nodeType() == ASTNodeType::Load) {
            usesRAW.emplace(d.later().as<ExprNode>(),
                            d.earlier().as<StmtNode>());
        } else if (d.earlier()->nodeType() == ASTNodeType::Load) {
            usesWAR.emplace(d.later().as<StmtNode>(),
                            d.earlier().as<ExprNode>());
        } else {
            ASSERT(false);
        }
    };

    findDeps(op, {{}}, foundOverwrite, FindDepsMode::Kill, DEP_WAW,
             filterOverwrite, false);
    findDeps(op, {{}}, foundUse, FindDepsMode::Dep, DEP_WAR | DEP_RAW, nullptr,
             false);

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
            for (auto &&use : usesRAW) {
                if (use.second == _earlier &&
                    usesWAR.count(std::make_pair(_later, use.first))) {
                    return;
                }
            }
            for (auto &&use : overwrites) {
                if (use.second == _earlier && !redundant.count(use.first) &&
                    overwrites.count(std::make_pair(_later, use.first))) {
                    // In the case of:
                    // (1) A = X
                    // (2) A += Y
                    // (3) A += Z
                    // if we handle (1)-(3) first, which resulting to A = X + Z,
                    // we cannot handle (2) then. So, we handle (1)-(2) before
                    // (1)-(3)
                    visitType1(use);
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
                // FIXME: What if StoreNode::expr_ is modified between the
                // StoreNode and the ReduceNode?
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
    auto variantExpr = findLoopVariance(op);
    FindLoopInvariantWrites finder(variantExpr);
    finder(op);
    for (auto &&item : finder.results()) {
        auto &&store = item.first.as<StmtNode>();
        auto &&cond = item.second;
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
    return flattenStmtSeq(op);
}

} // namespace ir

