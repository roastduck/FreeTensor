#ifndef FREE_TENSOR_ALL_STMTS_H
#define FREE_TENSOR_ALL_STMTS_H

#include <unordered_set>

#include <visitor.h>

namespace freetensor {

class AllStmts : public Visitor {
    const std::unordered_set<ASTNodeType> &types_;
    std::vector<Stmt> results_;

  public:
    AllStmts(const std::unordered_set<ASTNodeType> &types) : types_(types) {}

    const std::vector<Stmt> &results() { return results_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

/**
 * Record all statements of some specific node types
 *
 * Returns in program order
 */
inline std::vector<Stmt>
allStmts(const Stmt &op, const std::unordered_set<ASTNodeType> &types) {
    AllStmts visitor(types);
    visitor(op);
    return visitor.results();
}

} // namespace freetensor

#endif // FREE_TENSOR_ALL_STMTS_H
