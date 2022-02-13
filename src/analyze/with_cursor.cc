#include <analyze/with_cursor.h>

namespace ir {

void GetCursorById::visitStmt(const Stmt &op) {
    if (!found_) {
        WithCursor<Visitor>::visitStmt(op);
        if (op->id() == id_) {
            result_ = cursor();
            result_.push(op);
            ASSERT(result_.id() == id_);
            found_ = true;
        }
    }
}

void GetCursorByFilter::visitStmt(const Stmt &op) {
    WithCursor<Visitor>::visitStmt(op);
    auto c = cursor();
    c.push(op);
    if (filter_(c)) {
        results_.emplace_back(c);
    }
}

} // namespace ir
