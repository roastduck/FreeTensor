#include <unordered_map>

#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <pass/flatten_stmt_seq.h>

namespace ir {

static std::unordered_map<std::string, ID>
usedDefsAt(const Stmt &ast, const ID &pos,
           const std::unordered_set<std::string> &names) {
    CheckNameToDefMapping checker(pos, names);
    checker(ast);
    return checker.name2def();
}

void CheckNameToDefMapping::visitStmt(const Stmt &stmt) {
    BaseClass::visitStmt(stmt);
    if (stmt->id() == pos_) {
        for (auto &&name : names_) {
            if (hasDef(name)) {
                name2def_[name] = def(name)->id();
            } else if (hasLoop(name)) {
                name2def_[name] = loop(name)->id();
            }
        }
    }
}

Stmt InsertTmpEval::visitStmt(const Stmt &_op) {
    auto op = Mutator::visitStmt(_op);
    auto ret = op;
    if (op->id() == s0_) {
        auto eval = makeEval("", s0Expr_);
        s0Eval_ = eval->id();
        ret = s0Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq("", {eval, ret})
                  : makeStmtSeq("", {ret, eval});
    }
    if (op->id() == s1_) {
        auto eval = makeEval("", s1Expr_);
        s1Eval_ = eval->id();
        ret = s1Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq("", {eval, ret})
                  : makeStmtSeq("", {ret, eval});
    }
    return ret;
}

bool checkNotModified(const Stmt &op, const Expr &s0Expr, const Expr &s1Expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1) {
    auto names = allUses(s0Expr); // uses of s1 should be the same
    if (names.empty()) {
        return true;
    }
    if (usedDefsAt(op, s0, names) != usedDefsAt(op, s1, names)) {
        return false;
    }

    auto reads = allReads(s0Expr); // uses of s1 should be the same
    if (reads.empty()) {
        return true; // early exit: impossible to be written
    }

    // First insert temporarily Eval node to the AST, then perform dependency
    // analysis

    InsertTmpEval inserter(s0Expr, s1Expr, s0Side, s0, s1Side, s1);
    auto tmpOp = inserter(op);
    tmpOp = flattenStmtSeq(tmpOp);
    ASSERT(inserter.s0Eval().isValid());
    ASSERT(inserter.s1Eval().isValid());

    auto s0Eval = findStmt(tmpOp, inserter.s0Eval());
    auto s1Eval = findStmt(tmpOp, inserter.s1Eval());
    if (s0Eval->nextStmt() == s1Eval) {
        return true; // early exit: the period to check is empty
    }

    auto common = lcaStmt(s0Eval, s1Eval);
    FindDepsCond cond;
    for (auto p = common; p.isValid(); p = p->parentStmt()) {
        if (p->nodeType() == ASTNodeType::For) {
            cond.emplace_back(p->id(), DepDirection::Same);
        }
    }

    // write -> serialized PBSet
    std::unordered_map<Stmt, std::string> writesWAR;
    auto filterWAR = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.stmt_->id() == inserter.s0Eval();
    };
    auto foundWAR = [&](const Dependency &dep) {
        // Serialize dep.dep_ because it is from a random PBCtx
        writesWAR[dep.later_.stmt_] =
            toString(apply(domain(dep.dep_), dep.pmap_));
    };
    findDeps(tmpOp, {cond}, foundWAR, FindDepsMode::Dep, DEP_WAR, filterWAR,
             true, true, true);

    for (auto &&[_item, w0] : writesWAR) {
        auto &&item = _item;
        std::string w1;
        auto filterRAW = [&](const AccessPoint &later,
                             const AccessPoint &earlier) {
            return later.stmt_->id() == inserter.s1Eval() &&
                   earlier.stmt_->id() == item->id();
        };
        auto foundRAW = [&](const Dependency &dep) {
            // Serialize dep.dep_ because it is from a random PBCtx
            w1 = toString(apply(range(dep.dep_), dep.omap_));
        };
        findDeps(tmpOp, {cond}, foundRAW, FindDepsMode::Dep, DEP_RAW, filterRAW,
                 true, true, true);

        if (!w1.empty()) {
            PBCtx ctx;
            auto w = intersect(PBSet(ctx, w0), PBSet(ctx, w1));
            if (!w.empty()) {
                return false;
            }
        }
    }

    return true;
}

bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1) {
    return checkNotModified(op, expr, expr, s0Side, s0, s1Side, s1);
}

} // namespace ir
