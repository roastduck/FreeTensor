#include <analyze/deps.h>
#include <pass/simplify.h>
#include <schedule.h>
#include <schedule/blend.h>

namespace freetensor {

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
            throw InvalidSchedule("The length of " + toString(loop_) +
                                  " should be a constant");
        }
        iter_ = op->iter_;
        begin_ = op->begin_;
        step_ = op->step_;
        len_ = op->len_.as<IntConstNode>()->val_;
        inLoop_ = true;
        auto body = (*this)(op->body_);
        inLoop_ = false;
        return body;
    } else {
        if (inLoop_) {
            Stmt ret;
            if (!envStack_.empty() ||
                isVariant(exprVari_, {op->len_, op}, loop_)) {
                envStack_.emplace_back(op);
                ret = (*this)(op->body_);
                envStack_.pop_back();
            } else {
                ASSERT(!offset_.count(op->iter_));
                offset_[op->iter_] = std::make_pair(op->begin_, op->step_);
                ret = (*this)(op->body_);
                offset_.erase(op->iter_);
                auto len = (*this)(op->len_);
                ret = makeFor(op->iter_, makeIntConst(0), len, makeIntConst(1),
                              len, op->property_, std::move(ret),
                              op->metadata(), op->id());
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
        if (!envStack_.empty() ||
            isVariant(exprVari_, {op->cond_, op}, loop_)) {
            envStack_.emplace_back(makeIf(op->cond_, op->thenCase_));
            thenCase = (*this)(op->thenCase_);
            envStack_.pop_back();
            if (op->elseCase_.isValid()) {
                envStack_.emplace_back(
                    makeIf(makeLNot(op->cond_), op->elseCase_));
                elseCase = (*this)(op->elseCase_);
                envStack_.pop_back();
                return makeStmtSeq({thenCase, elseCase});
            } else {
                return thenCase;
            }
        } else {
            thenCase = (*this)(op->thenCase_);
            if (op->elseCase_.isValid()) {
                elseCase = (*this)(op->elseCase_);
                return makeIf((*this)(op->cond_), std::move(thenCase),
                              std::move(elseCase), op->metadata(), op->id());
            } else {
                return makeIf((*this)(op->cond_), std::move(thenCase),
                              op->metadata(), op->id());
            }
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const Assert &op) {
    if (inLoop_) {
        Stmt ret;
        if (!envStack_.empty() ||
            isVariant(exprVari_, {op->cond_, op}, loop_)) {
            envStack_.emplace_back(op);
            ret = Mutator::visit(op);
            envStack_.pop_back();
        } else {
            ret = Mutator::visit(op);
            ret = makeAssert((*this)(op->cond_), std::move(ret), op->metadata(),
                             op->id());
        }
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt BlendPass::visit(const VarDef &op) {
    if (inLoop_ && isVariant(varVari_, op, loop_)) {
        std::vector<Expr> shape;
        shape.reserve(op->buffer_->tensor()->shape().size());
        for (auto &&dim : op->buffer_->tensor()->shape()) {
            shape.emplace_back((*this)(dim));
        }
        Ref<Tensor> t =
            makeTensor(std::move(shape), op->buffer_->tensor()->dtype());
        Ref<Buffer> b = makeBuffer(std::move(t), op->buffer_->atype(),
                                   op->buffer_->mtype());

        defs_.emplace_back(op);
        auto body = (*this)(op->body_);
        defs_.pop_back();

        for (int k = len_ - 1; k >= 0; k--) {
            body = makeVarDef(op->name_ + "." + std::to_string(k), std::move(b),
                              op->viewOf_, std::move(body), op->pinned_);
        }
        return body;
    } else {
        return Mutator::visit(op);
    }
}

Expr BlendPass::visit(const Var &op) {
    if (inLoop_ && op->name_ == iter_) {
        return makeAdd(begin_, makeMul(makeIntConst(curIter_), step_));
    }
    if (offset_.count(op->name_)) {
        auto &&[begin, step] = offset_.at(op->name_);
        return makeAdd(makeMul(Mutator::visit(op), (*this)(step)),
                       (*this)(begin));
    }
    return Mutator::visit(op);
}

Expr BlendPass::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return visitMemAccess(op);
}

Stmt blend(const Stmt &_ast, const ID &loop) {
    auto ast =
        simplify(_ast); // Make things like range(n, n + 4) constant ranges

    FindAllScopesInside finder(loop);
    finder(ast);
    if (!finder.found()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }

    std::vector<FindDepsDir> direction;
    for (auto &&item : finder.scopes()) {
        direction.push_back(
            {{loop, DepDirection::Normal}, {item, DepDirection::Inv}});
    }
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() == 2);
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    FindDeps().direction(direction).filterSubAST(loop)(ast, found);

    auto loopVari = findLoopVariance(ast);
    ast = BlendPass(loop, loopVari.first, loopVari.second)(ast);
    return ast;
}

void Schedule::blend(const ID &loop) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(Blend, freetensor::blend, loop));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
