#ifndef FIND_ALL_SCOPES_H
#define FIND_ALL_SCOPES_H

#include <visitor.h>

namespace ir {

class FindAllScopes : public Visitor {
    std::vector<std::string> scopes_;

  public:
    const std::vector<std::string> &scopes() const { return scopes_; }

  protected:
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

/**
 * Collect IDs of all For and StmtSeq nodes in the AST
 */
std::vector<std::string> findAllScopes(const Stmt &op);

} // namespace ir

#endif // FIND_ALL_SCOPES_H
