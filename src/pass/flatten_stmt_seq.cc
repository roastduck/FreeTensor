#include <pass/flatten_stmt_seq.h>

namespace freetensor {

bool FlattenStmtSeq::isEmptySeq(const Stmt &s) {
    return s->nodeType() == ASTNodeType::StmtSeq &&
           s.as<StmtSeqNode>()->stmts_.empty();
}

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

Stmt FlattenStmtSeq::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (isEmptySeq(op->body_) && !isOutputting(op->buffer_->atype())) {
        return makeStmtSeq({});
    }
    return op;
}

Stmt FlattenStmtSeq::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (isEmptySeq(op->body_)) {
        return makeStmtSeq({});
    }
    return op;
}

Stmt FlattenStmtSeq::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (op->elseCase_.isValid() && isEmptySeq(op->elseCase_)) {
        op->elseCase_ = nullptr;
    }
    if (op->elseCase_.isValid()) {
        if (isEmptySeq(op->thenCase_)) {
            op->cond_ = makeLNot(op->cond_);
            op->thenCase_ = op->elseCase_;
            op->elseCase_ = nullptr;
        }
    } else {
        if (isEmptySeq(op->thenCase_)) {
            return makeStmtSeq({});
        }
    }
    return op;
}

Stmt FlattenStmtSeq::visit(const Assert &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (isEmptySeq(op->body_)) {
        return makeStmtSeq({});
    }
    return op;
}

Stmt FlattenStmtSeq::visit(const Assume &op) { return (*this)(op->body_); }

} // namespace freetensor
