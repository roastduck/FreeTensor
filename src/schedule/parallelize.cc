#include <schedule/parallelize.h>

namespace ir {

Stmt Parallelize::visit(const For &_op) {
    loopStack_.emplace_back(_op->id());
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    if (op->id() == loop_) {
        op->parallel_ = parallel_;
        outerLoops_ = loopStack_;
        done_ = true;
    }
    return op;
}

} // namespace ir
