#include <analyze/all_stmts.h>

namespace ir {

void AllStmts::visitStmt(const Stmt &op) {
    if (types_.count(op->nodeType())) {
        results_.emplace_back(op);
    }
    Visitor::visitStmt(op);
}

} // namespace ir
