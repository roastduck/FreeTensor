#include <unordered_set>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>

namespace ir {

Stmt InsertTmpEval::visitStmt(
    const Stmt &op, const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = Mutator::visitStmt(op, visitNode);
    if (ret->id() == s0_) {
        auto eval = makeEval("", expr_);
        s0Eval_ = eval->id();
        return makeStmtSeq("", {ret, eval});
    }
    if (ret->id() == s1_) {
        auto eval = makeEval("", expr_);
        s1Eval_ = eval->id();
        return makeStmtSeq("", {eval, ret});
    }
    return ret;
}

bool checkNotModified(const Stmt &op, const Expr &expr, const std::string &s0,
                      const std::string &s1) {
    // First insert temporarily Eval node to the AST, then perform dependency
    // analysis

    InsertTmpEval inserter(expr, s0, s1);
    auto tmpOp = inserter(op);

    std::unordered_set<Stmt> writesWAR, writesRAW;
    auto filterWAR = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.id() == inserter.s0Eval();
    };
    auto filterRAW = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.cursor_.id() == inserter.s1Eval();
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
