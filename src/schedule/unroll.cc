#include <schedule/unroll.h>

namespace ir {

Stmt Unroll::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            op->unroll_ = true;
            done_ = true;
        } else {
            throw InvalidSchedule("Length of the loop should be constant.");
        }
    }
    return op;
}

} // namespace ir
