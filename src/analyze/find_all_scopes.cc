#include <analyze/find_all_scopes.h>

namespace ir {

void FindAllScopes::visit(const For &op) {
    Visitor::visit(op);
    scopes_.emplace_back(op->id());
}

void FindAllScopes::visit(const StmtSeq &op) {
    Visitor::visit(op);
    scopes_.emplace_back(op->id());
}

std::vector<ID> findAllScopes(const Stmt &op) {
    FindAllScopes visitor;
    visitor(op);
    return visitor.scopes();
}

} // namespace ir
