#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <hash.h>
#include <pass/prop_const.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_dead_var.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <schedule/fuse.h>

namespace ir {

namespace {

std::vector<std::string> intersect(const std::vector<std::string> &lhs,
                                   const std::vector<std::string> &rhs) {
    std::vector<std::string> ret;
    for (auto &&item : lhs) {
        if (std::find(rhs.begin(), rhs.end(), item) != rhs.end()) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

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
            auto fused = makeFor(
                fused_, iter0_, makeIntConst(0), loop0->end_, makeIntConst(1),
                loop0->end_,
                ForProperty().withNoDeps(intersect(loop0->property_.noDeps_,
                                                   loop1->property_.noDeps_)),
                std::move(seq));

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

std::pair<Stmt, std::string> fuse(const Stmt &_ast, const std::string &loop0,
                                  const std::string &loop1, bool strict) {
    FuseFor mutator(loop0, loop1, strict);
    CheckAccessible check(loop0, loop1);
    check(_ast);
    if (!check.loop0().loop_.isValid()) {
        throw InvalidSchedule("Loops not found in a StmtSeq");
    }

    for (auto &&stmt : check.loop1().surroundings_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            for (auto shape :
                 stmt.as<VarDefNode>()->buffer_->tensor().shape()) {
                if (!checkNotModified(_ast, shape, CheckNotModifiedSide::Before,
                                      loop0, CheckNotModifiedSide::Before,
                                      loop1)) {
                    throw InvalidSchedule(
                        (std::string) "The shape of Vars in loop1 shouldn't "
                                      "be changed in loop0");
                }
            }
        }
    }

    auto ast = mutator(_ast);

    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.getParentById(mutator.afterId()).isValid() &&
               later.cursor_.getParentById(mutator.beforeId()).isValid();
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
        throw InvalidSchedule((std::string) "Fusing " + loop0 + " and " +
                              loop1 + " loop1 with different lengths? " +
                              e.what());
    }

    ast = propOneTimeUse(ast);
    ast = propConst(ast);
    ast = sinkVar(ast);
    ast = shrinkVar(ast);
    ast = removeDeadVar(ast);
    return std::make_pair(ast, mutator.fused());
}

} // namespace ir
