#include <unordered_set>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>

namespace ir {

Stmt InsertTmpEval::visitStmt(const Stmt &_op) {
    auto op = Mutator::visitStmt(_op);
    auto ret = op;
    if (op->id() == s0_) {
        auto eval = makeEval("", expr_);
        s0Eval_ = eval->id();
        ret = s0Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq("", {eval, ret})
                  : makeStmtSeq("", {ret, eval});
    }
    if (op->id() == s1_) {
        auto eval = makeEval("", expr_);
        s1Eval_ = eval->id();
        ret = s1Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq("", {eval, ret})
                  : makeStmtSeq("", {ret, eval});
    }
    return ret;
}

bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const std::string &s0,
                      CheckNotModifiedSide s1Side, const std::string &s1) {
    // First insert temporarily Eval node to the AST, then perform dependency
    // analysis

    InsertTmpEval inserter(expr, s0Side, s0, s1Side, s1);
    auto tmpOp = inserter(op);
    ASSERT(!inserter.s0Eval().empty());
    ASSERT(!inserter.s1Eval().empty());
    auto common = lca(getCursorById(tmpOp, inserter.s0Eval()),
                      getCursorById(tmpOp, inserter.s1Eval()));

    std::unordered_set<Stmt> writesWAR, writesRAW;
    auto filterWAR = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.id() == inserter.s0Eval() &&
               lca(later.cursor_, common).id() == common.id();
    };
    auto filterRAW = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.cursor_.id() == inserter.s1Eval() &&
               lca(earlier.cursor_, common).id() == common.id();
    };
    auto foundWAR = [&](const Dependency &dep) {
        writesWAR.insert(dep.later_.cursor_.node());
    };
    auto foundRAW = [&](const Dependency &dep) {
        writesRAW.insert(dep.earlier_.cursor_.node());
    };
    findDeps(tmpOp, {{}}, foundWAR, FindDepsMode::Dep, DEP_WAR, filterWAR);
    findDeps(tmpOp, {{}}, foundRAW, FindDepsMode::Dep, DEP_RAW, filterRAW);

    for (auto &item : writesWAR) {
        if (writesRAW.count(item)) {
            return false;
        }
    }

    // FIXME: What if the loop iterators are different between
    // `earlier` and `later`?

    return true;
}

} // namespace ir
