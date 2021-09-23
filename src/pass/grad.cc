#include <pass/grad.h>

namespace ir {

Expr ReplaceVar::visit(const Var &op) {
    if (op->name_ == from_) {
        return to_;
    } else {
        return Mutator::visit(op);
    }
}

void Grad::visit(const StmtSeq &op) {
    Visitor::visit(op);
    std::vector<Stmt> oriStmts, gradStmts;
    oriStmts.reserve(op->stmts_.size());
    gradStmts.reserve(op->stmts_.size() * 2);
    for (auto it = op->stmts_.begin(); it != op->stmts_.end(); it++) {
        if (oriStmts_.count(*it)) {
            oriStmts.emplace_back(oriStmts_.at(*it));
            gradStmts.emplace_back(oriStmts_.at(*it));
        }
    }
    for (auto it = op->stmts_.rbegin(); it != op->stmts_.rend(); it++) {
        if (gradStmts_.count(*it)) {
            gradStmts.emplace_back(gradStmts_.at(*it));
        }
    }
    oriStmts_[op] = makeStmtSeq("", std::move(oriStmts));
    gradStmts_[op] = makeStmtSeq("", std::move(gradStmts));
}

void Grad::visit(const For &op) {
    Visitor::visit(op);
    if (oriStmts_.count(op->body_)) {
        oriStmts_[op] = makeFor("", op->iter_, op->begin_, op->end_, op->len_,
                                op->noDeps_, op->parallel_, op->unroll_,
                                op->vectorize_, oriStmts_.at(op->body_));
    }
    if (gradStmts_.count(op->body_)) {
        gradStmts_[op] = makeFor(
            "", op->iter_, op->begin_, op->end_, op->len_, op->noDeps_,
            op->parallel_, op->unroll_, op->vectorize_,
            ReplaceVar(op->iter_, makeSub(op->end_, makeVar(op->iter_)))(
                gradStmts_.at(op->body_)));
    }
}

void Grad::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    Visitor::visit(op);
    buffers_.erase(op->name_);

    auto grad = gradStmts_.at(op->body_);
    if (gradNames_.count(op->name_)) {
        auto &&gradName = gradNames_.at(op->name_);

        if (op->buffer_->atype() != AccessType::Output &&
            op->buffer_->atype() != AccessType::InOut) {
            std::vector<std::string> iters;
            std::vector<Expr> indices;
            int nDim = op->buffer_->tensor().shape().size();
            iters.reserve(nDim);
            indices.reserve(nDim);
            for (int i = 0; i < nDim; i++) {
                std::string iter = "." + gradName + ".i" + std::to_string(i);
                indices.emplace_back(makeVar(iter));
                iters.emplace_back(std::move(iter));
            }
            auto init =
                makeStore("", gradName, std::move(indices), makeIntConst(0));
            for (int i = nDim - 1; i >= 0; i--) {
                init = makeFor("", iters[i], makeIntConst(0),
                               op->buffer_->tensor().shape()[i],
                               op->buffer_->tensor().shape()[i], false, "",
                               false, false, init);
            }
            grad = makeStmtSeq("", {init, grad});
        }

        grad = makeVarDef("", gradName, *op->buffer_, op->sizeLim_, grad,
                          op->pinned_);
        grad = makeVarDef(op->id(), op->name_, *op->buffer_, op->sizeLim_, grad,
                          op->pinned_);

        switch (op->buffer_->atype()) {
        case AccessType::Input:
            grad.as<VarDefNode>()->body_.as<VarDefNode>()->buffer_->setAtype(
                AccessType::Output);
            break;
        case AccessType::Output:
        case AccessType::InOut:
            grad.as<VarDefNode>()->buffer_->setAtype(AccessType::Input);
            grad.as<VarDefNode>()->body_.as<VarDefNode>()->buffer_->setAtype(
                AccessType::InOut);
            break;
        case AccessType::Cache:
            break; // do nothing
        }
    }
    gradStmts_[op] = grad;
}

void Grad::visit(const Store &op) {
    auto &&buffer = buffers_.at(op->var_);
    if (buffer->atype() == AccessType::Cache) {
        oriStmts_[op] = op;
    }

    std::vector<Stmt> stmts;
    if (gradNames_.count(op->var_)) {
        // Gradient of y[i] = f(x[i], y[i]) is:
        // d_y.old = d_y[i]
        // d_y[i] = 0
        // deduce d_x[i] and d_y[i] using d_y.old

        auto &&grad = gradNames_.at(op->var_);
        auto oldGrad = grad + ".old";
        auto &&indices = op->indices_;
        stmts.emplace_back(makeStore("", oldGrad, {}, makeLoad(grad, indices)));
        stmts.emplace_back(makeStore("", grad, indices, makeIntConst(0)));
        gradExprs_[op->expr_] = makeLoad(oldGrad, {});

        Visitor::visit(op);

        for (auto &&stmt : appends_) {
            stmts.emplace_back(stmt);
        }
        appends_.clear();
        // FIXME: Use a optimal MemType
        gradStmts_[op] =
            makeVarDef("", oldGrad,
                       Buffer(Tensor({}, buffer->tensor().dtype()),
                              AccessType::Cache, buffer->mtype()),
                       nullptr, makeStmtSeq("", std::move(stmts)), false);
    }
}

void Grad::visit(const Load &op) {
    Visitor::visit(op);
    if (gradExprs_.count(op) && gradNames_.count(op->var_)) {
        appends_.push_back(makeReduceTo("", gradNames_.at(op->var_),
                                        op->indices_, ReduceOp::Add,
                                        gradExprs_.at(op), false));
    }
}

void Grad::visit(const Add &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = gradExprs_[op->rhs_] = gradExprs_.at(op);
    }
    Visitor::visit(op);
}

void Grad::visit(const Sub &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = gradExprs_.at(op);
        gradExprs_[op->rhs_] = makeSub(makeIntConst(0), gradExprs_.at(op));
    }
    Visitor::visit(op);
}

void Grad::visit(const Mul &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = makeMul(gradExprs_.at(op), op->rhs_);
        gradExprs_[op->rhs_] = makeMul(gradExprs_.at(op), op->lhs_);
    }
    Visitor::visit(op);
}

void Grad::visit(const RealDiv &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = makeRealDiv(gradExprs_.at(op), op->rhs_);
        gradExprs_[op->rhs_] = makeSub(
            makeIntConst(0), makeRealDiv(makeMul(gradExprs_.at(op), op->lhs_),
                                         makeMul(op->rhs_, op->rhs_)));
    }
    Visitor::visit(op);
}

void Grad::visit(const Min &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = makeIfExpr(makeLE(op->lhs_, op->rhs_),
                                          gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->rhs_] = makeIfExpr(makeLT(op->rhs_, op->lhs_),
                                          gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void Grad::visit(const Max &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = makeIfExpr(makeGE(op->lhs_, op->rhs_),
                                          gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->rhs_] = makeIfExpr(makeGT(op->rhs_, op->lhs_),
                                          gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void Grad::visit(const IfExpr &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->thenCase_] =
            makeIfExpr(op->cond_, gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->elseCase_] =
            makeIfExpr(makeLNot(op->cond_), gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void Grad::visit(const Sqrt &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] =
            makeRealDiv(gradExprs_.at(op), makeMul(makeIntConst(2), op));
    }
    Visitor::visit(op);
}

void Grad::visit(const Exp &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] = makeMul(gradExprs_.at(op), op);
    }
    Visitor::visit(op);
}

void Grad::visit(const Square &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] =
            makeMul(makeIntConst(2), makeMul(gradExprs_.at(op), op->expr_));
    }
    Visitor::visit(op);
}

} // namespace ir
