#include <pass/flatten_stmt_seq.h>

namespace ir {

Stmt FlattenStmtSeq::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    for (auto &&item : op->stmts_) {
        if (item->nodeType() == ASTNodeType::StmtSeq) {
            for (auto &&subitem : item.as<StmtSeqNode>()->stmts_) {
                stmts.emplace_back(subitem);
            }
        } else {
            stmts.emplace_back(item);
        }
    }
    if (stmts.size() == 1) {
        return stmts[0];
    } else {
        return makeStmtSeq(op->id(), std::move(stmts));
    }
}

} // namespace ir

