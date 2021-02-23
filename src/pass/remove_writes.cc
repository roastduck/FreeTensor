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
    op = prepareFindDeps(op);

    // {(later, earlier)}
    std::set<std::pair<Stmt, Stmt>> overwrites;
    std::set<std::pair<Expr, Stmt>> usesRAW;
    std::set<std::pair<Stmt, Expr>> usesWAR;
    auto foundOverwrite = [&](const Dependency &d) {
        if (d.later()->nodeType() == ASTNodeType::Store ||
            sameParent(d.later_.cursor_, d.earlier_.cursor_)) {
            overwrites.emplace(d.later().as<StmtNode>(),
                               d.earlier().as<StmtNode>());
        }
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

    findDeps(op, {{}}, foundOverwrite, FindDepsMode::Kill, DEP_WAW, false);
    findDeps(op, {{}}, foundUse, FindDepsMode::Dep, DEP_WAR | DEP_RAW, false);

    std::unordered_set<Stmt> redundant;
    std::unordered_map<Stmt, Stmt> replacement;

    // Type 1
    for (auto &&item : overwrites) {
        auto &&later = item.first, &&earlier = item.second;
        for (auto &&use : usesRAW) {
            if (use.second == earlier &&
                usesWAR.count(std::make_pair(later, use.first))) {
                goto type1Fail;
            }
        }
        if (later->nodeType() == ASTNodeType::Store) {
            redundant.insert(earlier);
        } else {
            ASSERT(later->nodeType() == ASTNodeType::ReduceTo);
            // FIXME: What if StoreNode::expr_ is modified between the StoreNode
            // and the ReduceNode?
            auto l = later.as<ReduceToNode>();
            if (earlier->nodeType() == ASTNodeType::Store) {
                redundant.insert(earlier);
                replacement.emplace(
                    later,
                    makeStore(later->id(), l->var_, l->indices_,
                              makeReduce(l->op_, earlier.as<StoreNode>()->expr_,
                                         l->expr_)));
            } else if (earlier.as<ReduceToNode>()->op_ == l->op_) {
                redundant.insert(earlier);
                replacement.emplace(
                    later,
                    makeReduceTo(later->id(), l->var_, l->indices_, l->op_,
                                 makeReduce(l->op_,
                                            earlier.as<ReduceToNode>()->expr_,
                                            l->expr_),
                                 false));
            }
        }
        continue;
    type1Fail:;
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

