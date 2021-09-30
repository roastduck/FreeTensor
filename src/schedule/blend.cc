#include <schedule/blend.h>

namespace ir {

void FindAllScopesInside::visit(const For &op) {
    if (op->id() == loop_) {
        found_ = true;
        inLoop_ = true;
        Visitor::visit(op);
        inLoop_ = false;
    } else {
        if (inLoop_) {
            scopes_.emplace_back(op->id());
        }
        Visitor::visit(op);
    }
}

void FindAllScopesInside::visit(const StmtSeq &op) {
    if (inLoop_) {
        scopes_.emplace_back(op->id());
    }
    Visitor::visit(op);
}

Stmt BlendPass::visit(const For &op) {
    if (op->id() == loop_) {
        if (op->len_->nodeType() != ASTNodeType::IntConst) {
            throw InvalidSchedule("The length of " + loop_ +
                                  " should be a constant");
        }
        iter_ = op->iter_;
        begin_ = op->begin_;
        len_ = op->len_.as<IntConstNode>()->val_;
        inLoop_ = true;
        auto body = (*this)(op->body_);
        inLoop_ = false;
        return body;
    } else {
        if (inLoop_) {
            Stmt ret;
            if (!envStack_.empty() || isVariant(exprVari_, op->len_, loop_)) {
                envStack_.emplace_back(op);
                ret = (*this)(op->body_);
                envStack_.pop_back();
            } else {
                ASSERT(!offset_.count(op->iter_));
                offset_[op->iter_] = op->begin_;
                ret = (*this)(op->body_);
                offset_.erase(op->iter_);
                auto len = (*this)(op->len_);
                ret = makeFor(op->id(), op->iter_, makeIntConst(0), len, len,
                              op->noDeps_, op->parallel_, op->unroll_,
                              op->vectorize_, std::move(ret));
            }
            return ret;
        } else {
            return Mutator::visit(op);
        }
    }
}

Stmt BlendPass::visit(const If &op) {
    if (inLoop_) {
        Stmt thenCase, elseCase;
        if (!envStack_.empty() || isVariant(exprVari_, op->cond_, loop_)) {
            envStack_.emplace_back(makeIf("", op->cond_, op->thenCase_));
            thenCase = (*this)(op->thenCase_);
            envStack_.pop_back();
            if (op->elseCase_.isValid()) {
                envStack_.emplace_back(
                    makeIf("", makeLNot(op->cond_), op->elseCase_));
                elseCase = (*this)(op->elseCase_);
                envStack_.pop_back();
                return makeStmtSeq("", {thenCase, elseCase});
            } else {
                return thenCase;
            }
        } else {
            thenCase = (*this)(op->thenCase_);
            if (op->elseCase_.isValid()) {
                elseCase = (*this)(op->elseCase_);
                return makeIf(op->id(), (*this)(op->cond_), std::move(thenCase),
                              std::move(elseCase));
            } else {
                return makeIf(op->id(), (*this)(op->cond_),
                              std::move(thenCase));
            }
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const Assert &op) {
    if (inLoop_) {
        Stmt ret;
        if (!envStack_.empty() || isVariant(exprVari_, op->cond_, loop_)) {
            envStack_.emplace_back(op);
            ret = Mutator::visit(op);
            envStack_.pop_back();
        } else {
            ret = Mutator::visit(op);
            ret = makeAssert(op->id(), (*this)(op->cond_), std::move(ret));
        }
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const VarDef &op) {
    if (inLoop_ && isVariant(varVari_, op, loop_)) {
        defs_.emplace_back(op);
        auto body = (*this)(op->body_);
        auto sizeLim = op->sizeLim_.isValid() ? (*this)(op->sizeLim_) : nullptr;
        for (int k = len_ - 1; k >= 0; k--) {
            body =
                makeVarDef("", op->name_ + "." + std::to_string(k),
                           *op->buffer_, sizeLim, std::move(body), op->pinned_);
        }
        defs_.pop_back();
        return body;
    } else {
        return Mutator::visit(op);
    }
}

Expr BlendPass::visit(const Var &op) {
    if (inLoop_ && op->name_ == iter_) {
        return makeAdd(begin_, makeIntConst(curIter_));
    }
    if (offset_.count(op->name_)) {
        return makeAdd(Mutator::visit(op), (*this)(offset_.at(op->name_)));
    }
    return Mutator::visit(op);
}

Expr BlendPass::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return visitMemAccess(op);
}

} // namespace ir
