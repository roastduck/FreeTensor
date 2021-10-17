#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>

namespace ir {

namespace gpu {

void FindAllThreads::visit(const For &op) {
    if (op->property_.parallel_ == "threadIdx.x") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thx_ = op->len_.as<IntConstNode>()->val_;
    } else if (op->property_.parallel_ == "threadIdx.y") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thy_ = op->len_.as<IntConstNode>()->val_;
    } else if (op->property_.parallel_ == "threadIdx.z") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thz_ = op->len_.as<IntConstNode>()->val_;
    }
    Visitor::visit(op);
    if (op->property_.parallel_ == "threadIdx.x") {
        results_.emplace_back(ThreadInfo{op, thx_ <= warpSize_});
    } else if (op->property_.parallel_ == "threadIdx.y") {
        results_.emplace_back(ThreadInfo{op, thx_ * thy_ <= warpSize_});
    } else if (op->property_.parallel_ == "threadIdx.z") {
        results_.emplace_back(ThreadInfo{op, thx_ * thy_ <= warpSize_});
    }
}

Stmt CopyPart::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    if (!ended_ && end_.isValid() && op->id() == end_->id()) {
        ended_ = true;
    }
    if (ended_) {
        return makeStmtSeq("", {});
    }
    auto ret = Mutator::visitStmt(op, visitNode);
    if (!begun_) {
        ret = makeStmtSeq("", {});
    }
    if (!begun_ && begin_.isValid() && op->id() == begin_->id()) {
        begun_ = true;
    }
    return ret;
}

Stmt CopyPart::visit(const For &op) {
    bool begun = begun_, ended = ended_;
    auto ret = Mutator::visit(op);
    if (op->property_.parallel_.empty() &&
        ((!begun && begun_) || (!ended && ended_))) {
        throw InvalidProgram(
            "Unable to insert a synchronizing statment because it requires "
            "splitting loop " +
            op->id() + " into two parts");
    }
    return ret;
}

Stmt CopyPart::visit(const VarDef &_op) {
    bool begun = begun_, ended = ended_;
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if ((!begun && begun_) || (!ended && ended_)) {
        splittedDefs_.emplace_back(op);
        return op->body_;
    }
    return op;
}

Stmt MakeSync::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = MutatorWithCursor::visitStmt(op, visitNode);
    // Please note that we have exited MutatorWithCursor, so `cursor()` is out
    // of `op`

    Cursor target;
    bool needSyncThreads = false, needSyncWarp = false;
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.later_.node() == op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
            target =
                dep.lcaStmt_.depth() > target.depth() ? dep.lcaStmt_ : target;
        }
    }

    if (needSyncThreads || needSyncWarp) {
        Stmt sync;
        if (needSyncThreads) {
            sync = makeEval(
                "", makeIntrinsic("__syncthreads()", {}, DataType::Void));
        } else {
            sync =
                makeEval("", makeIntrinsic("__syncwarp()", {}, DataType::Void));
        }

        Cursor whereToInsert;
        for (auto ctx = cursor(); ctx.id() != target.id(); ctx = ctx.outer()) {
            if (ctx.nodeType() == ASTNodeType::For) {
                whereToInsert = ctx;
            }
        }
        if (!whereToInsert.isValid()) {
            ret = makeStmtSeq("", {sync, ret});
        } else {
            syncBeforeFor_[whereToInsert.id()] = sync;
        }

        bool inElseCase = false;
        for (auto ctx = whereToInsert.isValid() ? whereToInsert : cursor();
             ctx.hasOuter(); ctx = ctx.outer()) {
            if (ctx.nodeType() == ASTNodeType::If) {
                (inElseCase ? branchSplittersElse_
                            : branchSplittersThen_)[ctx.id()]
                    .emplace_back(sync);
            }
            inElseCase =
                ctx.outer().isValid() &&
                ctx.outer().nodeType() == ASTNodeType::If &&
                ctx.outer().node().as<IfNode>()->elseCase_ == ctx.node();
        }

        for (CrossThreadDep &dep : deps_) {
            if (dep.visiting_) {
                if (needSyncThreads) {
                    dep.synced_ = true;
                }
                if (needSyncWarp && dep.inWarp_) {
                    dep.synced_ = true;
                }
            }
        }
    }
    for (CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && !dep.visiting_ && dep.earlier_.node() == op) {
            dep.visiting_ = true;
        }
    }
    return ret;
}

Stmt MakeSync::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    bool needSyncThreads = false, needSyncWarp = false;
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.lcaLoop_.node() == _op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
        }
    }
    if (needSyncThreads) {
        op->body_ = makeStmtSeq(
            "", {op->body_, makeEval("", makeIntrinsic("__syncthreads()", {},
                                                       DataType::Void))});
    } else if (needSyncWarp) {
        op->body_ = makeStmtSeq(
            "", {op->body_, makeEval("", makeIntrinsic("__syncwarp()", {},
                                                       DataType::Void))});
    }
    if (needSyncThreads || needSyncWarp) {
        for (CrossThreadDep &dep : deps_) {
            if (dep.visiting_) {
                if (needSyncThreads) {
                    dep.synced_ = true;
                }
                if (needSyncWarp && dep.inWarp_) {
                    dep.synced_ = true;
                }
            }
        }
    }
    if (syncBeforeFor_.count(op->id())) {
        return makeStmtSeq("", {syncBeforeFor_.at(op->id()), op});
    }
    return op;
}

Stmt MakeSync::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();

    if (branchSplittersThen_.count(op->id()) ||
        branchSplittersElse_.count(op->id())) {
        if (!checkNotModified(root_, op->cond_, CheckNotModifiedSide::Before,
                              op->id(), CheckNotModifiedSide::After,
                              op->id())) {
            throw InvalidProgram("Unable to insert a synchronizing statment "
                                 "inside an If node because the condition " +
                                 toString(op->cond_) + " is being modified");
        }

        auto &&splittersThen = branchSplittersThen_[op->id()];
        auto &&splittersElse = branchSplittersElse_[op->id()];
        std::vector<Stmt> stmts;
        stmts.reserve(splittersThen.size() * 2 + 1);
        std::vector<VarDef> splittedDefs;
        for (size_t i = 0, iEnd = splittersThen.size() + 1; i < iEnd; i++) {
            Stmt begin = i == 0 ? nullptr : splittersThen[i - 1];
            Stmt end = i == iEnd - 1 ? nullptr : splittersThen[i];
            CopyPart copier(begin, end);
            auto part = copier(op->thenCase_);
            stmts.emplace_back(makeIf("", op->cond_, part));
            if (i < iEnd - 1) {
                stmts.emplace_back(splittersThen[i]);
            }
            for (auto &&item : copier.splittedDefs()) {
                if (std::find_if(splittedDefs.begin(), splittedDefs.end(),
                                 [&](const VarDef &x) {
                                     return x->id() == item->id();
                                 }) == splittedDefs.end()) {
                    splittedDefs.emplace_back(item);
                }
            }
        }
        Stmt thenBody = makeStmtSeq("", std::move(stmts));
        for (auto &&def : splittedDefs) {
            // FIXME: Check the shape is invariant
            thenBody =
                makeVarDef(def->id(), def->name_, std::move(*def->buffer_),
                           def->sizeLim_, thenBody, def->pinned_);
        }

        if (op->elseCase_.isValid()) {
            std::vector<Stmt> stmts;
            stmts.reserve(splittersElse.size() * 2 + 1);
            std::vector<VarDef> splittedDefs;
            for (size_t i = 0, iEnd = splittersElse.size() + 1; i < iEnd; i++) {
                Stmt begin = i == 0 ? nullptr : splittersElse[i - 1];
                Stmt end = i == iEnd - 1 ? nullptr : splittersElse[i];
                CopyPart copier(begin, end);
                auto part = copier(op->elseCase_);
                stmts.emplace_back(makeIf("", makeLNot(op->cond_), part));
                if (i < iEnd - 1) {
                    stmts.emplace_back(splittersElse[i]);
                }
                for (auto &&item : copier.splittedDefs()) {
                    if (std::find_if(splittedDefs.begin(), splittedDefs.end(),
                                     [&](const VarDef &x) {
                                         return x->id() == item->id();
                                     }) == splittedDefs.end()) {
                        splittedDefs.emplace_back(item);
                    }
                }
            }
            Stmt elseBody = makeStmtSeq("", std::move(stmts));
            for (auto &&def : splittedDefs) {
                // FIXME: Check the shape is invariant
                elseBody =
                    makeVarDef(def->id(), def->name_, std::move(*def->buffer_),
                               def->sizeLim_, elseBody, def->pinned_);
            }
            return makeStmtSeq("", {thenBody, elseBody});
        } else {
            return thenBody;
        }
    }
    return op;
}

Stmt makeSync(const Stmt &_op) {
    auto op = _op;
    FindAllThreads finder;
    finder(op);
    auto &&threads = finder.results();

    std::vector<FindDepsCond> query;
    query.reserve(threads.size());
    for (auto &&thr : threads) {
        query.push_back({{thr.loop_->id(), DepDirection::Different}});
    }
    std::vector<CrossThreadDep> deps;
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        if (later.op_ == earlier.op_) {
            return false;
        }
        return later.buffer_->mtype() == MemType::GPUGlobal ||
               later.buffer_->mtype() == MemType::GPUShared;
    };
    auto found = [&](const Dependency &d) {
        auto i = &d.cond_ - &query[0];
        auto lcaStmt = lca(d.later_.cursor_, d.earlier_.cursor_);
        auto lcaLoop = lcaStmt;
        while (lcaLoop.hasOuter() && lcaLoop.nodeType() != ASTNodeType::For) {
            lcaLoop = lcaLoop.outer();
        }
        ASSERT(lcaLoop.nodeType() == ASTNodeType::For);
        deps.emplace_back(CrossThreadDep{d.later_.cursor_, d.earlier_.cursor_,
                                         lcaStmt, lcaLoop,
                                         threads.at(i).inWarp_, false, false});
    };
    findDeps(op, query, found, FindDepsMode::Dep, DEP_ALL, filter, false,
             false);

    MakeSync mutator(op, std::move(deps));
    op = mutator(op);

    return flattenStmtSeq(op);
}

} // namespace gpu

} // namespace ir

