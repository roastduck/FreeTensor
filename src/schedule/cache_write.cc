#include <schedule/cache_write.h>

namespace ir {

Stmt CacheWrite::visit(const Store &op) {
    return inside_ ? visitStoreLike(op) : doModify(op);
}

Stmt CacheWrite::visit(const AddTo &op) {
    return inside_ ? visitStoreLike(op) : doModify(op);
}

Stmt CacheWrite::visit(const VarDef &op) {
    if (op->name_ == var_) {
        buffer_ = op->buffer_;
    }
    return Mutator::visit(op);
}

} // namespace ir

