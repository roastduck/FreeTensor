#include <schedule/merge.h>

namespace ir {

Stmt MergeFor::visit(const For &_op) {
    if (_op->id_ == oldOuter_->id_) {
        insideOuter_ = true;
        auto __op = Mutator::visit(_op);
        insideOuter_ = false;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        return makeFor(newId_, newIter_, makeIntConst(0),
                       makeMul(innerLen_, outerLen_), op->body_);
    } else if (_op->id_ == oldInner_->id_) {
        insideInner_ = true;
        auto __op = Mutator::visit(_op);
        insideInner_ = false;
        visitedInner_ = true;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        return op->body_;
    } else {
        return Mutator::visit(_op);
    }
}

Stmt MergeFor::visit(const StmtSeq &_op) {
    if (insideOuter_) {
        if (insideInner_) {
            return Mutator::visit(_op);
        }

        Stmt before, inner, after;
        std::vector<Stmt> beforeStmts, afterStmts;
        for (auto &&_stmt : _op->stmts_) {
            bool beforeInner = !visitedInner_;
            auto stmt = (*this)(_stmt);
            bool afterInner = visitedInner_;
            bool isInner = beforeInner && afterInner;
            if (isInner) {
                inner = stmt;
            } else if (beforeInner) {
                beforeStmts.emplace_back(stmt);
            } else {
                ASSERT(afterInner);
                afterStmts.emplace_back(stmt);
            }
        }

        if (!beforeStmts.empty()) {
            before = makeIf(
                "",
                makeEQ(makeMod(makeVar(newIter_), innerLen_), makeIntConst(0)),
                beforeStmts.size() == 1 ? beforeStmts[0]
                                        : makeStmtSeq(beforeStmts));
        }
        if (!afterStmts.empty()) {
            after = makeIf(
                "",
                makeEQ(makeMod(makeVar(newIter_), innerLen_), makeIntConst(0)),
                afterStmts.size() == 1 ? afterStmts[0]
                                       : makeStmtSeq(afterStmts));
        }

        std::vector<Stmt> stmts;
        if (before.isValid()) {
            stmts.emplace_back(before);
        }
        if (inner.isValid()) {
            stmts.emplace_back(inner);
        }
        if (after.isValid()) {
            stmts.emplace_back(after);
        }
        return stmts.size() == 1 ? stmts[0] : makeStmtSeq(stmts);
    } else {
        return Mutator::visit(_op);
    }
}

Expr MergeFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (op->name_ == oldInner_->iter_) {
        return makeMod(makeVar(newIter_), innerLen_);
    }
    if (op->name_ == oldOuter_->iter_) {
        return makeDiv(makeVar(newIter_), innerLen_);
    }
    return op;
}

} // namespace ir

