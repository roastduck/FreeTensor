#include <analyze/hash.h>
#include <schedule/cache_read.h>

namespace ir {

Expr CacheRead::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (inside_ && op->var_ == var_) {
        auto h = getHash(_op);
        for (auto &&item : loads_) {
            if (h == item.first) {
                goto done;
            }
        }
        loads_.emplace_back(h, _op);
    done:
        op->var_ = cacheVar_;
    }
    return op;
}

} // namespace ir

