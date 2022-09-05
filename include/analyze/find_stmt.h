#ifndef FREE_TENSOR_FIND_STMT_H
#define FREE_TENSOR_FIND_STMT_H

#include <func.h>
#include <selector.h>
#include <visitor.h>

namespace freetensor {

class FindStmtById : public Visitor {
    ID id_;
    Stmt result_;

  public:
    FindStmtById(const ID &id) : id_(id) {}

    const Stmt &result() const { return result_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline Stmt findStmt(const Stmt &ast, const ID &id) {
    FindStmtById visitor(id);
    visitor(ast);
    if (!visitor.result().isValid()) {
        throw UnexpectedQueryResult("Statement " + toString(id) + " not found");
    }
    return visitor.result();
}
inline std::vector<Stmt> findAllStmt(const Stmt &ast, const ID &id) {
    FindStmtById visitor(id);
    visitor(ast);
    if (!visitor.result().isValid()) {
        return {};
    }
    return {visitor.result()};
}

class FindStmtByFilter : public Visitor {
    const std::function<bool(const Stmt &)> &filter_;
    std::vector<Stmt> results_;

  public:
    FindStmtByFilter(const std::function<bool(const Stmt &)> &filter)
        : filter_(filter) {}
    const std::vector<Stmt> &results() const { return results_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline std::vector<Stmt>
findAllStmt(const Stmt &ast, const std::function<bool(const Stmt &)> &filter) {
    FindStmtByFilter visitor(filter);
    visitor(ast);
    return visitor.results();
}
inline Stmt findStmt(const Stmt &ast,
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

inline std::vector<Stmt> findAllStmt(const Stmt &ast,
                                     const Ref<Selector> &selector) {
    return findAllStmt(
        ast, [&selector](const Stmt &c) { return selector->match(c); });
}
inline Stmt findStmt(const Stmt &ast, const Ref<Selector> &selector) {
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

template <class T>
std::vector<Stmt> findAllStmt(const Func &func, const T &filter) {
    return findAllStmt(func->body_, filter);
}
template <class T> Stmt findStmt(const Func &func, const T &filter) {
    return findStmt(func->body_, filter);
}

} // namespace freetensor

#endif // FREE_TENSOR_FIND_STMT_H
