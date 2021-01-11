#include <schedule/parallelize.h>

namespace ir {

Stmt Parallelize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        op->parallel_ = parallel_;
        done_ = true;
    }
    return op;
}

} // namespace ir

