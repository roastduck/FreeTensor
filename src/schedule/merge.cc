#include <schedule/check_loop_order.h>
#include <schedule/merge.h>

namespace ir {

static std::vector<std::string> intersect(const std::vector<std::string> &lhs,
                                          const std::vector<std::string> &rhs) {
    std::vector<std::string> ret;
    for (auto &&item : lhs) {
        if (std::find(rhs.begin(), rhs.end(), item) != rhs.end()) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

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
        auto ret = makeFor(newId_, newIter_, makeIntConst(0), len, len,
                           ForProperty().withNoDeps(
                               intersect(op->property_.noDeps_, innerNoDeps_)),
                           op->body_);
        for (auto &&def : intermediateDefs_) {
            ret = makeVarDef(def->id(), def->name_, *def->buffer_,
                             def->sizeLim_, ret, def->pinned_);
        }
        return ret;
    } else if (_op->id() == oldInner_->id()) {
        insideInner_ = true;
        auto __op = Mutator::visit(_op);
        insideInner_ = false;
        visitedInner_ = true;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        innerNoDeps_ = op->property_.noDeps_;
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
                                        : makeStmtSeq("", beforeStmts));
        }
        if (!afterStmts.empty()) {
            after = makeIf(
                "",
                makeEQ(makeMod(makeVar(newIter_), innerLen_), makeIntConst(0)),
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

Expr MergeFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (insideInner_ && op->name_ == oldInner_->iter_) {
        return makeMod(makeVar(newIter_), innerLen_);
    }
    if (insideOuter_ && op->name_ == oldOuter_->iter_) {
        return makeFloorDiv(makeVar(newIter_), innerLen_);
    }
    return op;
}

Stmt MergeFor::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (insideOuter_ && !insideInner_) {
        intermediateDefs_.emplace_back(op);
        return op->body_;
    } else {
        return op;
    }
}

std::pair<Stmt, std::string> merge(const Stmt &_ast, const std::string &loop1,
                                   const std::string &loop2) {

    CheckLoopOrder checker({loop1, loop2});
    checker(_ast); // Check they are nested
    auto &&curOrder = checker.order();
    auto outer = curOrder[0], inner = curOrder[1];

    MergeFor mutator(outer, inner);
    auto ast = mutator(_ast);
    return std::make_pair(ast, mutator.newId());
}

} // namespace ir
