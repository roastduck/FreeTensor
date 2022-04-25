#include <algorithm>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/merge_no_deps_hint.h>
#include <hash.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_dead_var.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/tensor_prop_const.h>
#include <schedule/fuse.h>

namespace ir {

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
    if (stmt->nodeType() == ASTNodeType::VarDef) {
        auto ret =
            findLoopInScopes(stmt.as<VarDefNode>()->body_, id, direction);
        ret.scopes_.emplace_back(stmt);
        return ret;
    } else if (stmt->nodeType() == ASTNodeType::If) {
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
    } else if (stmt->nodeType() == ASTNodeType::StmtSeq) {
        auto stmtSeq = stmt.as<StmtSeqNode>();
        if (!stmtSeq->stmts_.empty()) {
            LoopInScopes ret;
            if (direction == FindLoopInScopesDirection::Front) {
                ret = findLoopInScopes(stmtSeq->stmts_.front(), id, direction);
            } else {
                ret = findLoopInScopes(stmtSeq->stmts_.back(), id, direction);
            }
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
        return makeFor(op->id(), op->iter_, makeIntConst(0), op->len_,
                       makeIntConst(1), op->len_, op->property_, op->body_);
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
                    body0 =
                        makeIf(branch->id(), branch->cond_, std::move(body0));
                } else if (stmt->nodeType() == ASTNodeType::Assert) {
                    auto ass = stmt.as<AssertNode>();
                    body0 = makeAssert(ass->id(), ass->cond_, std::move(body0));
                }
            }
            for (auto &&stmt : loop1InScopes.scopes_) {
                if (stmt->nodeType() == ASTNodeType::If) {
                    auto branch = stmt.as<IfNode>();
                    body1 =
                        makeIf(branch->id(), branch->cond_, std::move(body1));
                } else if (stmt->nodeType() == ASTNodeType::Assert) {
                    auto ass = stmt.as<AssertNode>();
                    body1 = makeAssert(ass->id(), ass->cond_, std::move(body1));
                }
            }
            auto seq = makeStmtSeq("", {std::move(body0), std::move(body1)});

            auto fused = makeFor(fused_, iter0_, makeIntConst(0), loop0->end_,
                                 makeIntConst(1), loop0->end_,
                                 ForProperty().withNoDeps(mergeNoDepsHint(
                                     root_, loop0->id(), loop1->id())),
                                 std::move(seq));

            // From inner to outer
            for (auto &&stmt : loop1InScopes.scopes_) {
                if (stmt->nodeType() == ASTNodeType::VarDef) {
                    auto def = stmt.as<VarDefNode>();
                    fused = makeVarDef(def->id(), def->name_,
                                       std::move(def->buffer_), def->sizeLim_,
                                       fused, def->pinned_);
                } else if (stmt->nodeType() == ASTNodeType::StmtSeq) {
                    auto seq = stmt.as<StmtSeqNode>();
                    std::vector<Stmt> stmts = {fused};
                    stmts.insert(stmts.end(), seq->stmts_.begin() + 1,
                                 seq->stmts_.end());
                    fused = makeStmtSeq(seq->id(), std::move(stmts));
                }
            }
            for (auto &&stmt : loop0InScopes.scopes_) {
                if (stmt->nodeType() == ASTNodeType::VarDef) {
                    auto def = stmt.as<VarDefNode>();
                    fused = makeVarDef(def->id(), def->name_,
                                       std::move(def->buffer_), def->sizeLim_,
                                       fused, def->pinned_);
                } else if (stmt->nodeType() == ASTNodeType::StmtSeq) {
                    auto seq = stmt.as<StmtSeqNode>();
                    std::vector<Stmt> stmts(seq->stmts_.begin(),
                                            seq->stmts_.end() - 1);
                    stmts.emplace_back(fused);
                    fused = makeStmtSeq(seq->id(), std::move(stmts));
                }
            }

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
                    makeAssert("", makeEQ(loop0->end_, loop1->end_), fused);
            }
            op->stmts_.erase(op->stmts_.begin() + i + 1);
            break;
        }
    }
    return op;
}

void CheckAccessible::visit(const StmtSeq &op) {
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

std::pair<Stmt, ID> fuse(const Stmt &_ast, const ID &loop0, const ID &loop1,
                         bool strict) {
    CheckAccessible check(loop0, loop1);
    check(_ast);
    if (!check.loop0().loop_.isValid()) {
        throw InvalidSchedule("Loops not found in a StmtSeq");
    }

    for (auto &&stmt : check.loop1().scopes_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            for (auto &&shape :
                 stmt.as<VarDefNode>()->buffer_->tensor()->shape()) {
                if (!checkNotModified(_ast, shape, CheckNotModifiedSide::Before,
                                      loop0, CheckNotModifiedSide::Before,
                                      loop1)) {
                    throw InvalidSchedule("The shape of Vars in loop1 "
                                          "shouldn't be changed in loop0");
                }
            }
        }
    }

    FuseFor mutator(_ast, loop0, loop1, strict);
    auto ast = mutator(_ast);

    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.stmt_->ancestorById(mutator.afterId()).isValid() &&
               later.stmt_->ancestorById(mutator.beforeId()).isValid();
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    findDeps(ast, {{{mutator.fused(), DepDirection::Normal}}}, found,
             FindDepsMode::Dep, DEP_ALL, filter);

    try {
        ast = simplifyPass(ast);
    } catch (const AssertAlwaysFalse &e) {
        throw InvalidSchedule("Fusing " + toString(loop0) + " and " +
                              toString(loop1) +
                              " loop1 with different lengths? " + e.what());
    }

    ast = propOneTimeUse(ast);
    ast = tensorPropConst(ast);
    ast = sinkVar(ast);
    ast = shrinkVar(ast);
    ast = removeDeadVar(ast);
    return std::make_pair(ast, mutator.fused());
}

} // namespace ir
