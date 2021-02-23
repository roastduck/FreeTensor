#include <set>

#include <analyze/deps.h>
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
    for (auto &&item : overwrites) {
        auto &&later = item.first, &&earlier = item.second;
        for (auto &&use : usesRAW) {
            if (use.second == earlier &&
                usesWAR.count(std::make_pair(later, use.first))) {
                goto fail;
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
    fail:;
    }

    op = RemoveWrites(redundant, replacement)(op);
    return flattenStmtSeq(op);
}

} // namespace ir

