#include <pass/flatten_stmt_seq.h>

namespace freetensor {

Stmt FlattenStmtSeq::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    for (Stmt item : op->stmts_) {
        if (item->nodeType() == ASTNodeType::StmtSeq) {
            for (auto &&subitem : item.as<StmtSeqNode>()->stmts_) {
                stmts.emplace_back(subitem);
            }
        } else {
            stmts.emplace_back(item);
        }
    }

    return stmts.size() == 1
               ? stmts[0]
               : makeStmtSeq(std::move(stmts), op->metadata(), op->id());
}

Stmt FlattenStmtSeq::visit(const Assume &op) { return (*this)(op->body_); }

} // namespace freetensor
