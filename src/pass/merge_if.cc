#include <algorithm>

#include <analyze/all_reads.h>
#include <analyze/all_writes.h>
#include <analyze/hash.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/merge_if.h>

namespace ir {

Stmt MergeIf::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    for (auto &&stmt : op->stmts_) {
        if (!stmts.empty() && stmts.back()->nodeType() == ASTNodeType::If &&
            stmt->nodeType() == ASTNodeType::If) {
            auto if1 = stmts.back().as<IfNode>();
            auto if2 = stmt.as<IfNode>();
            if (getHash(if1->cond_) == getHash(if2->cond_)) {
                auto writes = allWrites(if1);
                auto reads = allReads(if2->cond_);
                if (std::none_of(reads.begin(), reads.end(),
                                [&writes](const std::string &var) -> bool {
                                    return writes.count(var);
                                })) {
                    auto thenCase =
                        makeStmtSeq("", {if1->thenCase_, if2->thenCase_});
                    Stmt elseCase;
                    if (if1->elseCase_.isValid() && if2->elseCase_.isValid()) {
                        elseCase =
                            makeStmtSeq("", {if1->elseCase_, if2->elseCase_});
                    } else if (if1->elseCase_.isValid() &&
                                !if2->elseCase_.isValid()) {
                        elseCase = if1->elseCase_;
                    } else if (!if1->elseCase_.isValid() &&
                                if2->elseCase_.isValid()) {
                        elseCase = if2->elseCase_;
                    }
                    stmts.pop_back();
                    stmts.emplace_back(makeIf("", if1->cond_,
                                            std::move(thenCase),
                                            std::move(elseCase)));
                    continue;
                }
            }
        }
        stmts.emplace_back(stmt);
    }
    op->stmts_ = std::move(stmts);
    return op;
}

Stmt mergeIf(const Stmt &op) {
    auto ret = MergeIf()(op);
    return flattenStmtSeq(ret);
}

} // namespace ir

