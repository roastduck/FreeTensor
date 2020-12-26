#include <algorithm>

#include <except.h>
#include <schedule/reorder.h>

namespace ir {

void CheckLoopOrder::visit(const For &op) {
    if (done_) {
        return;
    }
    if (!op->id_.empty() && std::find(dstOrder_.begin(), dstOrder_.end(),
                                      op->id_) != dstOrder_.end()) {
        curOrder_.emplace_back(op);
        if (curOrder_.size() < dstOrder_.size()) {
            Visitor::visit(op);
        }
        done_ = true;
        // done_ is to avoid such a program:
        // for i {
        //	 for j {}
        //	 for k {}
        // }
    } else {
        if (!curOrder_.empty()) { // Already met the first loop
            throw InvalidSchedule(
                "Unable to find all the loops to be reordered. "
                "These loops should be directly nested");
        }
        Visitor::visit(op);
    }
}

const std::vector<For> &CheckLoopOrder::order() const {
    if (curOrder_.size() != dstOrder_.size()) {
        throw InvalidSchedule("Unable to find all the loops to be reordered. "
                              "These loops should be directly nested");
    }
    return curOrder_;
}

Stmt SwapFor::visit(const For &_op) {
    if (_op->id_ == oldOuter_->id_) {
        insideOuter_ = true;
        auto body = Mutator::visit(_op);
        insideOuter_ = false;
        return makeFor(oldInner_->iter_, oldInner_->begin_, oldInner_->end_,
                       body, oldInner_->id_);
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

Stmt SwapFor::visit(const StmtSeq &_op) {
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
            before =
                makeIf(makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
                       beforeStmts.size() == 1 ? beforeStmts[0]
                                               : makeStmtSeq(beforeStmts));
        }
        if (!afterStmts.empty()) {
            after = makeIf(makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
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

} // namespace ir

