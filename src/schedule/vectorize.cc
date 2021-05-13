#include <schedule/vectorize.h>

namespace ir {

Stmt Vectorize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        op->vectorize_ = true;
        done_ = true;
    }
    return op;
}

} // namespace ir
