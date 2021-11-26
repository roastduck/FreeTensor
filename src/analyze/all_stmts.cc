#include <analyze/all_stmts.h>

namespace ir {

void AllStmts::visitStmt(const Stmt &op,
                         const std::function<void(const Stmt &)> &visitNode) {
    if (types_.count(op->nodeType())) {
        results_.emplace_back(op);
    }
    Visitor::visitStmt(op, visitNode);
}

} // namespace ir

