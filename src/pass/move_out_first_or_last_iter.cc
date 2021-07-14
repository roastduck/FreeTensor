#include <pass/move_out_first_or_last_iter.h>

namespace ir {

Expr MoveOutFirstOrLastIter::visit(const Var &op) {
    if (replace_.count(op->name_)) {
        return (*this)(replace_.at(op->name_));
    } else {
        return Z3Simplify::visit(op);
    }
}

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
                    ASSERT(!replace_.count(op->iter_));
                    replace_[op->iter_] = op->begin_;
                    toFront = (*this)(branch->thenCase_);
                    replace_.erase(op->iter_);
                    seq->stmts_.erase(seq->stmts_.begin());
                }
            }
            if (seq->stmts_.back()->nodeType() == ASTNodeType::If) {
                auto &&branch = seq->stmts_.back().as<IfNode>();
                if (!branch->elseCase_.isValid() &&
                    prove((*this)(
                        makeEQ(branch->cond_,
                               makeEQ(makeVar(op->iter_),
                                      makeSub(op->end_, makeIntConst(1))))))) {
                    ASSERT(!replace_.count(op->iter_));
                    replace_[op->iter_] = makeSub(op->end_, makeIntConst(1));
                    toBack = (*this)(branch->thenCase_);
                    replace_.erase(op->iter_);
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
