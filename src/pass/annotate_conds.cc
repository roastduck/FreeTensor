#include <analyze/all_uses.h>
#include <analyze/as_dnf.h>
#include <container_utils.h>
#include <pass/annotate_conds.h>

namespace freetensor {

void AnnotateConds::addCond(const Expr &expr) {
    auto dnf = asDNF(expr);
    if (dnf.size() == 1) {
        for (auto &&item : dnf.front()) {
            conds_.emplace_back(item);
        }
    } else {
        conds_.emplace_back(expr);
    }
}

Stmt AnnotateConds::visit(const StmtSeq &op) {
    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    for (auto &&stmt : op->stmts_) {
        auto &&writes = allWrites(stmt);
        Expr annotate;
        for (auto &&cond : conds_) {
            if (cond.isValid() && hasIntersect(allReads(cond), writes)) {
                // do not reset cond here, because there is finer-grained
                // structure in stmt
                annotate = annotate.isValid() ? makeLAnd(annotate, cond) : cond;
            }
        }
        if (annotate.isValid()) {
            auto assume =
                makeAssume("", annotate, makeStmtSeq("", std::move(stmts)));
            stmts = {assume};
        }
        stmts.emplace_back((*this)(stmt));
    }
    return makeStmtSeq(op->id(), std::move(stmts));
}

Stmt AnnotateConds::visit(const For &op) {
    auto &&writes = allWrites(op->body_);
    for (Expr &cond : conds_) {
        if (cond.isValid() && hasIntersect(allReads(cond), writes)) {
            cond = nullptr;
        }
    }

    auto begin = (*this)(op->begin_);
    auto end = (*this)(op->end_);
    auto step = (*this)(op->step_);
    auto len = (*this)(op->len_);

    auto oldCondsSize = conds_.size();
    auto var = makeVar(op->iter_);
    if (op->step_->nodeType() == ASTNodeType::IntConst) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            conds_.emplace_back(makeGE(var, op->begin_));
            conds_.emplace_back(makeLT(var, op->end_));
            conds_.emplace_back(makeEQ(
                makeMod(makeSub(var, op->begin_), op->step_), makeIntConst(0)));
        } else if (step < 0) {
            conds_.emplace_back(makeLE(var, op->begin_));
            conds_.emplace_back(makeGT(var, op->end_));
            conds_.emplace_back(makeEQ(
                makeMod(makeSub(var, op->begin_), op->step_), makeIntConst(0)));
        } else {
            conds_.emplace_back(makeEQ(var, op->begin_));
        }
    }
    auto body = (*this)(op->body_);
    conds_.resize(oldCondsSize);

    return makeFor(op->id(), op->iter_, std::move(begin), std::move(end),
                   std::move(step), std::move(len), op->property_,
                   std::move(body));
}

Stmt AnnotateConds::visit(const If &op) {
    auto cond = (*this)(op->cond_);

    auto oldCondsSize = conds_.size();
    addCond(op->cond_);
    auto thenCase = (*this)(op->thenCase_);
    conds_.resize(oldCondsSize);

    Stmt elseCase;
    if (op->elseCase_.isValid()) {
        oldCondsSize = conds_.size();
        addCond(makeLNot(op->cond_));
        elseCase = (*this)(op->elseCase_);
        conds_.resize(oldCondsSize);
    }

    return makeIf(op->id(), std::move(cond), std::move(thenCase),
                  std::move(elseCase));
}

Stmt AnnotateConds::visit(const Assert &op) {
    auto cond = (*this)(op->cond_);

    auto oldCondsSize = conds_.size();
    addCond(op->cond_);
    auto body = (*this)(op->body_);
    conds_.resize(oldCondsSize);

    return makeAssert(op->id(), std::move(cond), std::move(body));
}

Stmt AnnotateConds::visit(const Assume &op) { return (*this)(op->body_); }

} // namespace freetensor
