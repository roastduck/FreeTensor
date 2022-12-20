#include <analyze/merge_no_deps_hint.h>
#include <pass/tensor_prop_const.h>
#include <schedule.h>
#include <schedule/check_loop_order.h>
#include <schedule/hoist_selected_var.h>
#include <schedule/merge.h>

namespace freetensor {

Stmt MergeFor::visit(const For &_op) {
    if (_op->id() == oldOuter_->id()) {
        insideOuter_ = true;
        auto __op = Mutator::visit(_op);
        insideOuter_ = false;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        auto len = innerLen_->nodeType() == ASTNodeType::IntConst &&
                           outerLen_->nodeType() == ASTNodeType::IntConst
                       ? makeIntConst(innerLen_.as<IntConstNode>()->val_ *
                                      outerLen_.as<IntConstNode>()->val_)
                       : makeMul(innerLen_, outerLen_);
        auto ret =
            makeFor(newIter_, makeIntConst(0), len, makeIntConst(1), len,
                    Ref<ForProperty>::make()->withNoDeps(mergeNoDepsHint(
                        root_, oldInner_->id(), oldOuter_->id())),
                    op->body_, makeMetadata("merge", oldOuter_, oldInner_));
        newId_ = ret->id();
        return ret;
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
                makeEQ(makeMod(makeVar(newIter_), innerLen_), makeIntConst(0)),
                beforeStmts.size() == 1 ? beforeStmts[0]
                                        : makeStmtSeq(beforeStmts));
        }
        if (!afterStmts.empty()) {
            after = makeIf(
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
        return stmts.size() == 1
                   ? stmts[0]
                   : makeStmtSeq(stmts, _op->metadata(), _op->id());
    } else {
        return Mutator::visit(_op);
    }
}

Expr MergeFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (insideInner_ && op->name_ == oldInner_->iter_) {
        return makeAdd(
            oldInner_->begin_,
            makeMul(makeMod(makeVar(newIter_), innerLen_), oldInner_->step_));
    }
    if (insideOuter_ && op->name_ == oldOuter_->iter_) {
        return makeAdd(oldOuter_->begin_,
                       makeMul(makeFloorDiv(makeVar(newIter_), innerLen_),
                               oldOuter_->step_));
    }
    return op;
}

Stmt MergeFor::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (insideOuter_ && !insideInner_) {
        return op->body_;
    } else {
        return op;
    }
}

std::pair<Stmt, ID> merge(const Stmt &_ast, const ID &loop1, const ID &loop2) {
    // Propagate first, because merge will lose some propagating opportunities
    auto ast = tensorPropConst(_ast);

    CheckLoopOrder checker({loop1, loop2});
    checker(ast); // Check they are nested
    auto &&curOrder = checker.order();
    auto outer = curOrder[0], inner = curOrder[1];

    // Hoist VarDef nodes between `outer` and `inner` to out of `outer`
    ast = hoistSelectedVar(ast, "<<-" + toString(outer->id()) + "&->>" +
                                    toString(inner->id()));

    MergeFor mutator(ast, outer, inner);
    ast = mutator(ast);
    return std::make_pair(ast, mutator.newId());
}

ID Schedule::merge(const ID &loop1, const ID &loop2) {
    beginTransaction();
    auto log =
        appendLog(MAKE_SCHEDULE_LOG(Merge, freetensor::merge, loop1, loop2));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
