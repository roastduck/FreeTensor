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

/**
 * Find all statements from an AST by ID, filter or selector
 *
 * @return : All statements satisfying the given condition, in DFS order
 *
 * @{
 */
std::vector<Stmt> findAllStmt(const Stmt &ast, const ID &id);
std::vector<Stmt> findAllStmt(const Stmt &ast,
                              const std::function<bool(const Stmt &)> &filter);
std::vector<Stmt> findAllStmt(const Stmt &ast, const Ref<Selector> &selector);
template <class T>
std::vector<Stmt> findAllStmt(const Func &func, const T &filter) {
    return findAllStmt(func->body_, filter);
}
/** @} */

/**
 * Find the only statement from an AST by ID, filter or selector
 *
 * @return : The only statement
 * @throw UnexpectedQueryResult if zero or more than one statements are found
 *
 * @{
 */
Stmt findStmt(const Stmt &ast, const ID &id);
Stmt findStmt(const Stmt &ast, const std::function<bool(const Stmt &)> &filter);
Stmt findStmt(const Stmt &ast, const Ref<Selector> &selector);
template <class T> Stmt findStmt(const Func &func, const T &filter) {
    return findStmt(func->body_, filter);
}
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_FIND_STMT_H
