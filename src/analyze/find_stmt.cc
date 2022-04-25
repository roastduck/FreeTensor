#include <analyze/find_stmt.h>

namespace ir {

void FindStmtById::visitStmt(const Stmt &op) {
    if (!result_.isValid()) {
        Visitor::visitStmt(op);
        if (op->id() == id_) {
            result_ = op;
        }
    }
}

void FindStmtByFilter::visitStmt(const Stmt &op) {
    Visitor::visitStmt(op);
    if (filter_(op)) {
        results_.emplace_back(op);
    }
}

} // namespace ir
