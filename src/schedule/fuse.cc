#include <algorithm>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <analyze/merge_no_deps_hint.h>
#include <hash.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_dead_var.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/tensor_prop_const.h>
#include <schedule.h>
#include <schedule/fuse.h>
#include <schedule/hoist_selected_var.h>

namespace freetensor {

namespace {

LoopInScopes findLoopInScopes(const Stmt &stmt, const ID &id,
                              FindLoopInScopesDirection direction) {
    if (stmt->id() == id) {
        if (stmt->nodeType() != ASTNodeType::For) {
            throw InvalidSchedule("Statement " + toString(id) +
                                  " is not a loop");
        }
        return LoopInScopes{stmt.as<ForNode>(), {}};
    }
    if (stmt->nodeType() == ASTNodeType::If) {
        if (auto branch = stmt.as<IfNode>(); !branch->elseCase_.isValid()) {
            auto ret = findLoopInScopes(branch->thenCase_, id, direction);
            // Currently we don't support StmtSeq-in-If cases (TODO)
            if (std::find_if(ret.scopes_.begin(), ret.scopes_.end(),
                             [](const Stmt &s) {
                                 return s->nodeType() == ASTNodeType::StmtSeq;
                             }) == ret.scopes_.end()) {
                ret.scopes_.emplace_back(stmt);
                return ret;
            }
        }
    } else if (stmt->nodeType() == ASTNodeType::Assert) {
        auto ass = stmt.as<AssertNode>();
        auto ret = findLoopInScopes(ass->body_, id, direction);
        // Currently we don't support StmtSeq-in-Assert cases (TODO)
        if (std::find_if(ret.scopes_.begin(), ret.scopes_.end(),
                         [](const Stmt &s) {
                             return s->nodeType() == ASTNodeType::StmtSeq;
                         }) == ret.scopes_.end()) {
            ret.scopes_.emplace_back(stmt);
            return ret;
        }
    }
    return LoopInScopes{nullptr, {}};
}

} // Anonymous namespace

Expr FuseFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (inLoop0_ && op->name_ == iter0_) {
        return makeAdd(makeMul(makeVar(iter0_), step0_), begin0_);
    }
    if (inLoop1_ && op->name_ == iter1_) {
        // Yes, use iter0_
        return makeAdd(makeMul(makeVar(iter0_), step1_), begin1_);
    }
    return op;
}

Stmt FuseFor::visit(const For &_op) {
    if (_op->id() == id0_) {
        iter0_ = _op->iter_;
        begin0_ = _op->begin_, step0_ = _op->step_;
        inLoop0_ = true;
    }
    if (_op->id() == id1_) {
        iter1_ = _op->iter_;
        begin1_ = _op->begin_, step1_ = _op->step_;
        inLoop1_ = true;
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == id0_ || op->id() == id1_) {
        inLoop0_ = inLoop1_ = false;
        return makeFor(op->iter_, makeIntConst(0), op->len_, makeIntConst(1),
                       op->len_, op->property_, op->body_, op->metadata(),
                       op->id());
    }
    return op;
}

Stmt FuseFor::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        auto loop0InScopes = findLoopInScopes(op->stmts_[i], id0_,
                                              FindLoopInScopesDirection::Back);
        if (loop0InScopes.loop_.isValid()) {
            if (i + 1 == iEnd) {
                throw InvalidSchedule("Fuse: Loop " + toString(id0_) + " and " +
                                      toString(id1_) +
                                      " shuold be directly following");
            }
            auto loop1InScopes = findLoopInScopes(
                op->stmts_[i + 1], id1_, FindLoopInScopesDirection::Front);
            if (!loop1InScopes.loop_.isValid()) {
                throw InvalidSchedule("Fuse: Loop " + toString(id0_) + " and " +
                                      toString(id1_) +
                                      " shuold be directly following");
            }

            auto loop0 = loop0InScopes.loop_;
            auto loop1 = loop1InScopes.loop_;
            beforeId_ = loop0->body_->id();
            afterId_ = loop1->body_->id();

            Stmt body0 = loop0->body_;
            Stmt body1 = loop1->body_;
            // From inner to outer
            for (auto &&stmt : loop0InScopes.scopes_) {
                if (stmt->nodeType() == ASTNodeType::If) {
                    auto branch = stmt.as<IfNode>();
                    body0 = makeIf(branch->cond_, std::move(body0),
                                   branch->metadata(), branch->id());
                } else if (stmt->nodeType() == ASTNodeType::Assert) {
                    auto ass = stmt.as<AssertNode>();
                    body0 = makeAssert(ass->cond_, std::move(body0),
                                       ass->metadata(), ass->id());
                }
            }
            for (auto &&stmt : loop1InScopes.scopes_) {
                if (stmt->nodeType() == ASTNodeType::If) {
                    auto branch = stmt.as<IfNode>();
                    body1 = makeIf(branch->cond_, std::move(body1),
                                   branch->metadata(), branch->id());
                } else if (stmt->nodeType() == ASTNodeType::Assert) {
                    auto ass = stmt.as<AssertNode>();
                    body1 = makeAssert(ass->cond_, std::move(body1),
                                       ass->metadata(), ass->id());
                }
            }
            auto seq = makeStmtSeq({std::move(body0), std::move(body1)});

            auto fused =
                makeFor(iter0_, makeIntConst(0), loop0->end_, makeIntConst(1),
                        loop0->end_,
                        ForProperty().withNoDeps(
                            mergeNoDepsHint(root_, loop0->id(), loop1->id())),
                        std::move(seq), makeMetadata("fuse", loop0, loop1));
            fused_ = fused->id();

            if (strict_) {
                if (!HashComparator()(loop0->end_, loop1->end_)) {
                    throw InvalidSchedule(
                        "Unable to determine whether the two loops are of the "
                        "same length. If you are sure that they are the same, "
                        "please disable the strict mode");
                }
                op->stmts_[i] = fused;
            } else {
                op->stmts_[i] =
                    makeAssert(makeEQ(loop0->end_, loop1->end_), fused);
            }
            op->stmts_.erase(op->stmts_.begin() + i + 1);
            break;
        }
    }
    return op;
}

void CheckFuseAccessible::visit(const StmtSeq &op) {
    Visitor::visit(op);
    if (!loop0InScopes_.loop_.isValid()) {
        for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
            loop0InScopes_ = findLoopInScopes(op->stmts_[i], id0_,
                                              FindLoopInScopesDirection::Back);
            if (loop0InScopes_.loop_.isValid()) {
                if (i + 1 == iEnd) {
                    throw InvalidSchedule("Fuse: Loop " + toString(id0_) +
                                          " and " + toString(id1_) +
                                          " shuold be directly following");
                }
                loop1InScopes_ = findLoopInScopes(
                    op->stmts_[i + 1], id1_, FindLoopInScopesDirection::Front);
                if (!loop1InScopes_.loop_.isValid()) {
                    throw InvalidSchedule("Fuse: Loop " + toString(id0_) +
                                          " and " + toString(id1_) +
                                          " shuold be directly following");
                }
                return;
            }
        }
    }
}

void CheckFuseAccessible::check(const Stmt &ast) {
    (*this)(ast);

    if (!loop0().loop_.isValid()) {
        throw InvalidSchedule("Loops not found in a StmtSeq");
    }
}

std::pair<Stmt, ID> fuse(const Stmt &_ast, const ID &loop0, const ID &loop1,
                         bool strict) {
    // Hoist all VarDef nodes covering one of the loop but not covering the
    // other loop, to cover both loops
    auto ast = hoistSelectedVar(
        _ast, "(->>" + toString(loop0) + "&!->>" + toString(loop1) + ")|(!->>" +
                  toString(loop0) + "&->>" + toString(loop1) + ")");
    ast = flattenStmtSeq(ast);

    CheckFuseAccessible(loop0, loop1).check(ast);

    FuseFor mutator(ast, loop0, loop1, strict);
    ast = mutator(ast);

    FindDepsDir dir = {{mutator.fused(), DepDirection::Normal}};
    for (auto outer = findStmt(ast, mutator.fused())->parentStmt();
         outer.isValid(); outer = outer->parentStmt()) {
        if (outer->nodeType() == ASTNodeType::For) {
            dir.emplace_back(outer->id(), DepDirection::Same);
        }
    }
    FindDeps()
        .direction({dir})
        .filterEarlier([&](const AccessPoint &earlier) {
            return earlier.stmt_->ancestorById(mutator.afterId()).isValid();
        })
        .filterLater([&](const AccessPoint &later) {
            return later.stmt_->ancestorById(mutator.beforeId()).isValid();
        })(ast, [&](const Dependence &d) {
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        });

    try {
        ast = simplify(ast);
    } catch (const AssertAlwaysFalse &e) {
        throw InvalidSchedule("Fusing " + toString(loop0) + " and " +
                              toString(loop1) +
                              " loop1 with different lengths? " + e.what());
    }

    ast = propOneTimeUse(ast, mutator.fused());
    ast = tensorPropConst(ast, mutator.fused());
    ast = sinkVar(ast);
    ast = shrinkVar(ast);
    ast = removeDeadVar(ast);
    return std::make_pair(ast, mutator.fused());
}

ID Schedule::fuse(const ID &loop0, const ID &loop1, bool strict) {
    beginTransaction();
    auto log = appendLog(
        MAKE_SCHEDULE_LOG(Fuse, freetensor::fuse, loop0, loop1, strict));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

ID Schedule::fuse(const ID &loop0, bool strict) {
    beginTransaction();
    auto l0 = find(loop0);

    auto isTrivialScope = [](const Stmt &s) {
        switch (s->nodeType()) {
        case ASTNodeType::StmtSeq:
        case ASTNodeType::VarDef:
        case ASTNodeType::Assert:
        case ASTNodeType::Assume:
            return true;
        case ASTNodeType::If:
            return !s.as<IfNode>()->elseCase_.isValid();
        default:
            return false;
        }
    };
    auto firstStmtInTrivalScope = [](const Stmt &s) -> Stmt {
        switch (s->nodeType()) {
        case ASTNodeType::StmtSeq:
            return s.as<StmtSeqNode>()->stmts_.empty()
                       ? nullptr
                       : (Stmt)s.as<StmtSeqNode>()->stmts_.front();
        case ASTNodeType::VarDef:
            return s.as<VarDefNode>()->body_;
        case ASTNodeType::Assert:
            return s.as<AssertNode>()->body_;
        case ASTNodeType::Assume:
            return s.as<AssumeNode>()->body_;
        case ASTNodeType::If:
            return s.as<IfNode>()->elseCase_.isValid()
                       ? nullptr
                       : (Stmt)s.as<IfNode>()->thenCase_;
        default:
            return nullptr;
        }
    };

    auto s = l0;
    while (!s->nextStmt().isValid() && s->parentStmt().isValid() &&
           isTrivialScope(s->parentStmt())) {
        s = s->parentStmt();
    }
    if (s.isValid()) {
        if (s = s->nextStmt(); s.isValid()) {
            while (s.isValid() && s->nodeType() != ASTNodeType::For &&
                   isTrivialScope(s)) {
                s = firstStmtInTrivalScope(s);
            }
            if (s.isValid() && s->nodeType() == ASTNodeType::For) {
                auto ret = fuse(loop0, s->id(), strict);
                commitTransaction();
                return ret;
            }
        }
    }

    abortTransaction();
    throw InvalidSchedule("Invalid fuse(" + toString(loop0) +
                          "): Unable to find a following loop of " +
                          toString(loop0));
}

} // namespace freetensor
