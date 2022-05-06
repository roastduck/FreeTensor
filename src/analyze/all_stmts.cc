#include <analyze/all_stmts.h>

namespace freetensor {

void AllStmts::visitStmt(const Stmt &op) {
    if (types_.count(op->nodeType())) {
        results_.emplace_back(op);
    }
    Visitor::visitStmt(op);
}

} // namespace freetensor
