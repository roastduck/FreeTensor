#include <except.h>
#include <schedule/fuse.h>

namespace ir {

namespace {

LoopInVarDefs findLoopInVarDefs(const Stmt &stmt, const std::string &id,
                                FindLoopInVarDefsDirection direction) {
    if (stmt->id() == id) {
        if (stmt->nodeType() != ASTNodeType::For) {
            throw InvalidSchedule("Statement " + id + " is not a loop");
        }
        return LoopInVarDefs{stmt.as<ForNode>(), {}};
    }
    if (stmt->nodeType() == ASTNodeType::VarDef) {
        auto ret =
            findLoopInVarDefs(stmt.as<VarDefNode>()->body_, id, direction);
        ret.surroundings_.emplace_back(stmt);
        return ret;
    }
    if (stmt->nodeType() == ASTNodeType::StmtSeq) {
        auto stmtSeq = stmt.as<StmtSeqNode>();
        LoopInVarDefs ret;
        if (direction == FindLoopInVarDefsDirection::Front) {
            ret = findLoopInVarDefs(stmtSeq->stmts_.front(), id, direction);
        } else {
            ret = findLoopInVarDefs(stmtSeq->stmts_.back(), id, direction);
        }
        ret.surroundings_.emplace_back(stmt);
        return ret;
    }
    return LoopInVarDefs{nullptr, {}};
}

} // Anonymous namespace

Expr FuseFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (op->name_ == iter0_) {
        return makeAdd(makeVar(iter0_), begin0_);
    }
    if (op->name_ == iter1_) {
        // Yes, use iter0_
        return makeAdd(makeVar(iter0_), begin1_);
    }
    return op;
}

Stmt FuseFor::visit(const For &_op) {
    if (_op->id() == id0_) {
        iter0_ = _op->iter_, begin0_ = _op->begin_;
    }
    if (_op->id() == id1_) {
        iter1_ = _op->iter_, begin1_ = _op->begin_;
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == id0_ || op->id() == id1_) {
        return makeFor(op->id(), op->iter_, makeIntConst(0), op->len_, op->len_,
                       op->noDeps_, op->parallel_, op->unroll_, op->vectorize_,
                       op->body_);
    }
    return op;
}

Stmt FuseFor::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        auto loop0InVarDefs = findLoopInVarDefs(
            op->stmts_[i], id0_, FindLoopInVarDefsDirection::Back);
        if (loop0InVarDefs.loop_.isValid()) {
            if (i + 1 == iEnd) {
                throw InvalidSchedule("Fuse: Loop " + id0_ + " and " + id1_ +
                                      " shuold be directly following");
            }
            auto loop1InVarDefs = findLoopInVarDefs(
                op->stmts_[i + 1], id1_, FindLoopInVarDefsDirection::Front);
            if (!loop1InVarDefs.loop_.isValid()) {
                throw InvalidSchedule("Fuse: Loop " + id0_ + " and " + id1_ +
                                      " shuold be directly following");
            }

            auto loop0 = loop0InVarDefs.loop_;
            auto loop1 = loop1InVarDefs.loop_;
            beforeId_ = loop0->body_->id();
            afterId_ = loop1->body_->id();
            auto seq = makeStmtSeq("", {loop0->body_, loop1->body_});
            auto fused = makeFor(fused_, iter0_, makeIntConst(0), loop0->end_,
                                 loop0->end_, loop0->noDeps_ && loop1->noDeps_,
                                 loop0->parallel_, loop0->unroll_,
                                 loop0->vectorize_, std::move(seq));

            // From inner to outer
            for (auto &&stmt : loop1InVarDefs.surroundings_) {
                if (stmt->nodeType() == ASTNodeType::VarDef) {
                    auto def = stmt.as<VarDefNode>();
                    fused = makeVarDef(def->id(), def->name_,
                                       std::move(*def->buffer_), def->sizeLim_,
                                       fused, def->pinned_);
                } else {
                    auto seq = stmt.as<StmtSeqNode>();
                    std::vector<Stmt> stmts = {fused};
                    stmts.insert(stmts.end(), seq->stmts_.begin() + 1,
                                 seq->stmts_.end());
                    fused = makeStmtSeq(seq->id(), std::move(stmts));
                }
            }
            for (auto &&stmt : loop0InVarDefs.surroundings_) {
                if (stmt->nodeType() == ASTNodeType::VarDef) {
                    auto def = stmt.as<VarDefNode>();
                    fused = makeVarDef(def->id(), def->name_,
                                       std::move(*def->buffer_), def->sizeLim_,
                                       fused, def->pinned_);
                } else {
                    auto seq = stmt.as<StmtSeqNode>();
                    std::vector<Stmt> stmts(seq->stmts_.begin(),
                                            seq->stmts_.end() - 1);
                    stmts.emplace_back(fused);
                    fused = makeStmtSeq(seq->id(), std::move(stmts));
                }
            }

            op->stmts_[i] =
                makeAssert("", makeEQ(loop0->end_, loop1->end_), fused);
            op->stmts_.erase(op->stmts_.begin() + i + 1);
            break;
        }
    }
    return op;
}

void CheckAccessible::visit(const StmtSeq &op) {
    Visitor::visit(op);
    if (!loop0InVarDefs_.loop_.isValid()) {
        for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
            loop0InVarDefs_ = findLoopInVarDefs(
                op->stmts_[i], id0_, FindLoopInVarDefsDirection::Back);
            if (loop0InVarDefs_.loop_.isValid()) {
                if (i + 1 == iEnd) {
                    throw InvalidSchedule("Fuse: Loop " + id0_ + " and " +
                                          id1_ +
                                          " shuold be directly following");
                }
                loop1InVarDefs_ = findLoopInVarDefs(
                    op->stmts_[i + 1], id1_, FindLoopInVarDefsDirection::Front);
                if (!loop1InVarDefs_.loop_.isValid()) {
                    throw InvalidSchedule("Fuse: Loop " + id0_ + " and " +
                                          id1_ +
                                          " shuold be directly following");
                }
                return;
            }
        }
    }
}

} // namespace ir
