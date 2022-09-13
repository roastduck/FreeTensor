#include <analyze/find_stmt.h>

namespace freetensor {

void FindStmtById::visitStmt(const Stmt &op) {
    if (!result_.isValid()) {
        Visitor::visitStmt(op);
        if (op->id() == id_) {
            result_ = op;
        }
    }
}

void FindStmtByFilter::visitStmt(const Stmt &op) {
    if (filter_(op)) {
        results_.emplace_back(op);
        // Emplace before recursion: DFS pre order
    }
    Visitor::visitStmt(op);
}

Stmt findStmt(const Stmt &ast, const ID &id) {
    FindStmtById visitor(id);
    visitor(ast);
    if (!visitor.result().isValid()) {
        throw UnexpectedQueryResult("Statement " + toString(id) + " not found");
    }
    return visitor.result();
}

std::vector<Stmt> findAllStmt(const Stmt &ast, const ID &id) {
    FindStmtById visitor(id);
    visitor(ast);
    if (!visitor.result().isValid()) {
        return {};
    }
    return {visitor.result()};
}

std::vector<Stmt> findAllStmt(const Stmt &ast,
                              const std::function<bool(const Stmt &)> &filter) {
    FindStmtByFilter visitor(filter);
    visitor(ast);
    return visitor.results();
}

Stmt findStmt(const Stmt &ast,
              const std::function<bool(const Stmt &)> &filter) {
    auto candidates = findAllStmt(ast, filter);
    if (candidates.empty()) {
        throw UnexpectedQueryResult(
            "No statement found by filter. Consider using find_all_stmts or "
            "Schedule.find_all");
    }
    if (candidates.size() > 1) {
        throw UnexpectedQueryResult(
            "Multiple statements found by filter. Consider using "
            "find_all_stmts or Schedule.find_all");
    }
    return candidates.front();
}

std::vector<Stmt> findAllStmt(const Stmt &ast, const Ref<Selector> &selector) {
    return findAllStmt(
        ast, [&selector](const Stmt &c) { return selector->match(c); });
}

Stmt findStmt(const Stmt &ast, const Ref<Selector> &selector) {
    auto candidates = findAllStmt(ast, selector);
    if (candidates.empty()) {
        throw UnexpectedQueryResult(
            "No statement found by selector. Consider using find_all_stmts or "
            "Schedule.find_all");
    }
    if (candidates.size() > 1) {
        throw UnexpectedQueryResult(
            "Multiple statements found by selector. Consider using "
            "find_all_stmts or Schedule.find_all");
    }
    return candidates.front();
}

} // namespace freetensor
