#include <pass/move_out_first_or_last_iter.h>
#include <pass/replace_iter.h>

namespace ir {

Stmt MoveOutFirstOrLastIter::visit(const For &_op) {
    auto __op = Z3Simplify::visit(_op);
    if (__op->nodeType() == ASTNodeType::For) {
        auto op = __op.as<ForNode>();

        if (op->body_->nodeType() == ASTNodeType::StmtSeq) {
            Stmt toFront, toBack;
            auto &&seq = op->body_.as<StmtSeqNode>();
            if (seq->stmts_.front()->nodeType() == ASTNodeType::If) {
                auto &&branch = seq->stmts_.front().as<IfNode>();
                if (!branch->elseCase_.isValid() &&
                    prove(
                        (*this)(makeEQ(branch->cond_, makeEQ(makeVar(op->iter_),
                                                             op->begin_))))) {
                    toFront =
                        ReplaceIter(op->iter_, op->begin_)(branch->thenCase_);
                    toFront = (*this)(toFront);
                    seq->stmts_.erase(seq->stmts_.begin());
                }
            }
            if (seq->stmts_.back()->nodeType() == ASTNodeType::If) {
                auto &&branch = seq->stmts_.back().as<IfNode>();
                auto rbegin = makeAdd(
                    op->begin_,
                    makeMul(makeSub(op->len_, makeIntConst(1)), op->step_));
                if (!branch->elseCase_.isValid() &&
                    prove((*this)(makeEQ(
                        branch->cond_, makeEQ(makeVar(op->iter_), rbegin))))) {
                    toBack = ReplaceIter(op->iter_, rbegin)(branch->thenCase_);
                    toBack = (*this)(toBack);
                    seq->stmts_.pop_back();
                }
            }

            Stmt ret = op;
            if (toFront.isValid()) {
                ret = makeStmtSeq("", {toFront, ret});
            }
            if (toBack.isValid()) {
                ret = makeStmtSeq("", {ret, toBack});
            }
            return ret;
        }
    }
    return __op;
}

} // namespace ir
