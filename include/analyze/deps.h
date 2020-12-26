#ifndef DEPS_H
#define DEPS_H

#include <unordered_map>
#include <vector>

#include <mutator.h>
#include <visitor.h>

namespace ir {

typedef std::vector<Expr> AccessPoint;

/**
 * Find read and write points
 */
class FindAccessPoint : public Visitor {
    AccessPoint cur_; // Current iteration point in the space
    std::unordered_map<const ASTNode *, AccessPoint> reads_, writes_;

  public:
    const std::unordered_map<const ASTNode *, AccessPoint> &reads() const {
        return reads_;
    }
    const std::unordered_map<const ASTNode *, AccessPoint> &writes() const {
        return writes_;
    }

  protected:
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
};

/**
 * Find RAW, WAR and WAW dependencies, and mark them in the AST
 */
class AnalyzeDeps : public Mutator {
    const std::unordered_map<const ASTNode *, AccessPoint> &reads_, &writes_;

  public:
    AnalyzeDeps(const std::unordered_map<const ASTNode *, AccessPoint> &reads,
                const std::unordered_map<const ASTNode *, AccessPoint> &writes)
        : reads_(reads), writes_(writes) {}

  private:
    AccessPoint makeDep(const AccessPoint &lhs, const AccessPoint &rhs);

  protected:
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
};

Stmt analyzeDeps(const Stmt &op);

}; // namespace ir

#endif // DEPS_H
