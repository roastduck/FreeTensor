#ifndef ID_H
#define ID_H

#include <ast.h>

namespace ir {

/**
 * This is a general identifier for Expr OR Stmt
 *
 * An Stmt is identified by its `id()` function
 *
 * An Expr is identified by itself, and the `id()` of the Stmt it is in
 */
struct ExprOrStmtId {
    AST ast_;   /// The Expr or Stmt
    Expr expr_; /// null for Stmt
    std::string id_;

    ExprOrStmtId(const Stmt &stmt) : ast_(stmt), id_(stmt->id()) {}
    ExprOrStmtId(const Expr &expr, const Stmt &parent)
        : ast_(expr), expr_(expr), id_(parent->id()) {}
};

bool operator==(const ExprOrStmtId &lhs, const ExprOrStmtId &rhs);

} // namespace ir

namespace std {

template <> struct hash<ir::ExprOrStmtId> {
    size_t operator()(const ir::ExprOrStmtId &id) const;
};

} // namespace std

#endif // ID_H
