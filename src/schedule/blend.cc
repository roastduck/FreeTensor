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
        ASSERT(op->infoLen_.isValid());
        if (op->infoLen_->nodeType() != ASTNodeType::IntConst) {
            throw InvalidSchedule("The length of " + loop_ +
                                  " should be a constant");
        }
        iter_ = op->iter_;
        begin_ = op->begin_;
        len_ = op->infoLen_.as<IntConstNode>()->val_;
        inLoop_ = true;
        auto body = (*this)(op->body_);
        for (auto it = defs_.rbegin(); it != defs_.rend(); it++) {
            for (int k = len_ - 1; k >= 0; k--) {
                body = makeVarDef("", (*it)->name_ + "." + std::to_string(k),
                                  *(*it)->buffer_, (*it)->sizeLim_, body);
            }
        }
        inLoop_ = false;
        return body;
    } else {
        if (inLoop_) {
            envStack_.emplace_back(op);
            auto ret = (*this)(op->body_);
            envStack_.pop_back();
            return ret;
        } else {
            return Mutator::visit(op);
        }
    }
}

Stmt BlendPass::visit(const If &op) {
    if (inLoop_) {
        envStack_.emplace_back(makeIf("", op->cond_, op->thenCase_));
        auto thenCase = (*this)(op->thenCase_);
        envStack_.pop_back();

        if (op->elseCase_.isValid()) {
            envStack_.emplace_back(
                makeIf("", makeLNot(op->cond_), op->elseCase_));
            auto elseCase = (*this)(op->elseCase_);
            envStack_.pop_back();
            return makeStmtSeq("", {thenCase, elseCase});
        } else {
            return thenCase;
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const Assert &op) {
    if (inLoop_) {
        envStack_.emplace_back(op);
        auto ret = Mutator::visit(op);
        envStack_.pop_back();
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const VarDef &op) {
    if (inLoop_) {
        defs_.emplace_back(op);
        return (*this)(op->body_);
    } else {
        return Mutator::visit(op);
    }
}

Expr BlendPass::visit(const Var &op) {
    if (inLoop_ && op->name_ == iter_) {
        return makeAdd(begin_, makeIntConst(curIter_));
    } else {
        return Mutator::visit(op);
    }
}

Expr BlendPass::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return visitMemAccess(op);
}

} // namespace ir
