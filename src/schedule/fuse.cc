#include <except.h>
#include <schedule/fuse.h>

namespace ir {

Expr FuseFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (op->name_ == iter0_) {
        return makeAdd(makeVar(iter0_), begin0_);
    }
    if (op->name_ == iter1_) {
        // Yes, use iter0_
        return makeAdd(makeVar(iter0_), begin1_);
    }
    return op;
}

Stmt FuseFor::visit(const For &_op) {
    if (_op->id() == id0_) {
        iter0_ = _op->iter_, begin0_ = _op->begin_;
    }
    if (_op->id() == id1_) {
        iter1_ = _op->iter_, begin1_ = _op->begin_;
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == id0_ || op->id() == id1_) {
        op = makeFor(op->id(), op->iter_, makeIntConst(0),
                    makeSub(op->end_, op->begin_), op->parallel_, op->body_);
    }
    return op;
}

Stmt FuseFor::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        if (op->stmts_[i]->id() == id0_) {
            if (i + 1 == iEnd || op->stmts_[i + 1]->id() != id1_) {
                throw InvalidSchedule("Fuse: Loop " + id0_ + " and " + id1_ +
                                    " shuold be directly following");
            }
            auto loop0 = op->stmts_[i].as<ForNode>();
            auto loop1 = op->stmts_[i + 1].as<ForNode>();
            auto fused = makeFor(fused_, iter0_, makeIntConst(0), loop0->end_,
                                loop0->parallel_,
                                makeStmtSeq("", {loop0->body_, loop1->body_}));
            op->stmts_[i] =
                makeAssert("", makeEQ(loop0->end_, loop1->end_), fused);
            op->stmts_.erase(op->stmts_.begin() + i + 1);
            break;
        }
    }
    return op;
}

} // namespace ir

