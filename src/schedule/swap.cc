#include <algorithm>
#include <unordered_map>

#include <schedule/swap.h>

namespace ir {

Stmt Swap::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    std::unordered_map<std::string, size_t> pos;
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        pos[op->stmts_[i]->id()] = i;
    }
    size_t start = op->stmts_.size();
    for (auto &&item : order_) {
        if (!pos.count(item)) {
            return op;
        }
        start = std::min(start, pos.at(item));
    }

    auto stmts = op->stmts_;
    for (size_t i = 0, iEnd = order_.size(); i < iEnd; i++) {
        auto p = pos.at(order_[i]);
        if (p >= start + order_.size()) {
            return op;
        }
        stmts[start + i] = op->stmts_[p];
    }
    scope_ = _op; // the old one
    return makeStmtSeq(op->id(), std::move(stmts));
}

} // namespace ir

