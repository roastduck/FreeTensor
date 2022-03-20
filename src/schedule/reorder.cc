#include <sstream>

#include <analyze/deps.h>
#include <pass/make_reduction.h>
#include <schedule/check_loop_order.h>
#include <schedule/reorder.h>

namespace ir {

Stmt SwapFor::visit(const For &_op) {
    if (_op->id() == oldOuter_->id()) {
        insideOuter_ = true;
        auto body = Mutator::visit(_op);
        insideOuter_ = false;
        return makeFor(oldInner_->id(), oldInner_->iter_, oldInner_->begin_,
                       oldInner_->end_, oldInner_->step_, oldInner_->len_,
                       oldInner_->property_, body);
    } else if (_op->id() == oldInner_->id()) {
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
            if (!oldInner_->property_.parallel_.empty()) {
                throw InvalidSchedule("Imperfect nesting is not allowed when "
                                      "the inner loop is parallelized");
            }
            before =
                makeIf("", makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
                       beforeStmts.size() == 1 ? beforeStmts[0]
                                               : makeStmtSeq("", beforeStmts));
        }
        if (!afterStmts.empty()) {
            if (!oldInner_->property_.parallel_.empty()) {
                throw InvalidSchedule("Imperfect nesting is not allowed when "
                                      "the inner loop is parallelized");
            }
            after =
                makeIf("", makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
                       afterStmts.size() == 1 ? afterStmts[0]
                                              : makeStmtSeq("", afterStmts));
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
        return stmts.size() == 1 ? stmts[0] : makeStmtSeq(_op->id(), stmts);
    } else {
        return Mutator::visit(_op);
    }
}

Stmt reorder(const Stmt &_ast, const std::vector<ID> &dstOrder) {
    auto ast = makeReduction(_ast);

    CheckLoopOrder checker(dstOrder);
    checker(ast);
    auto curOrder = checker.order();

    std::vector<int> index;
    index.reserve(curOrder.size());
    for (auto &&loop : curOrder) {
        index.emplace_back(
            std::find(dstOrder.begin(), dstOrder.end(), loop->id()) -
            dstOrder.begin());
    }

    // A reorder is leagal if and only if, after transformation, there is no
    // such dependence that the out-most non-'=' carrying loop is not a '>'
    std::vector<FindDepsCond> conds;
    for (size_t i = 0, n = dstOrder.size(); i < n; i++) {
        FindDepsCond cond;
        for (size_t j = 0; j < i; j++) {
            cond.emplace_back(dstOrder[j], DepDirection::Same);
        }
        cond.emplace_back(dstOrder[i], DepDirection::Inv);
        conds.emplace_back(std::move(cond));
    }
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.getParentById(curOrder.front()->id())
                   .isValid() &&
               later.cursor_.getParentById(curOrder.front()->id()).isValid();
    };
    auto found = [&](const Dependency &d) {
        throw InvalidSchedule("Loops are not permutable: " + toString(d) +
                              " cannot be resolved");
    };
    findDeps(ast, conds, found, FindDepsMode::Dep, DEP_ALL, filter);

    // Bubble Sort
    size_t n = index.size();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j + 1 < n; j++) {
            if (index[j] > index[j + 1]) {
                SwapFor swapper(curOrder[j], curOrder[j + 1]);
                ast = swapper(ast);
                std::swap(index[j], index[j + 1]);
                std::swap(curOrder[j], curOrder[j + 1]);
            }
        }
    }

    return ast;
}

} // namespace ir
