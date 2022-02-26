#ifndef FIND_ALL_SCOPES_H
#define FIND_ALL_SCOPES_H

#include <visitor.h>

namespace ir {

class FindAllScopes : public Visitor {
    std::vector<ID> scopes_;

  public:
    const std::vector<ID> &scopes() const { return scopes_; }

  protected:
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

/**
 * Collect IDs of all For and StmtSeq nodes in the AST
 */
std::vector<ID> findAllScopes(const Stmt &op);

} // namespace ir

#endif // FIND_ALL_SCOPES_H
