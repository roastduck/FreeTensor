#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/gpu/make_sync.h>
#include <pass/merge_and_hoist_if.h>

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

Stmt CopyParts::visitStmt(const Stmt &op) {
    auto ret = Mutator::visitStmt(op);
    if (ret->nodeType() == ASTNodeType::Store ||
        ret->nodeType() == ASTNodeType::ReduceTo ||
        ret->nodeType() == ASTNodeType::Eval) {
        // leaf node
        if (std::find_if(splitters_.begin(), splitters_.end(),
                         [&](const Stmt &item) {
                             return item->id() == ret->id();
                         }) == splitters_.end()) {
            // not a splitter
            fullParts_.insert(ret);
        }
    }
    return ret;
}

Stmt CopyParts::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (fullParts_.count(op->body_)) {
        fullParts_.insert(op);
    } else {
        if (op->property_.parallel_.empty() &&
            (op->begin_->nodeType() != ASTNodeType::IntConst ||
             op->end_->nodeType() != ASTNodeType::IntConst)) {
            throw InvalidProgram(
                "Unable to insert a synchronizing statment because it requires "
                "splitting a dynamic loop " +
                op->id() + " into two parts");
        }
    }
    return op;
}

Stmt CopyParts::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (fullParts_.count(op->thenCase_) &&
        (!op->elseCase_.isValid() || fullParts_.count(op->elseCase_))) {
        fullParts_.insert(op);
    }
    return op;
}

Stmt CopyParts::visit(const Assert &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (fullParts_.count(op->body_)) {
        fullParts_.insert(op);
    }
    return op;
}

Stmt CopyParts::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (fullParts_.count(op->body_)) {
        fullParts_.insert(op);
    }
    return op;
}

Stmt CopyParts::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    for (auto &&stmt : op->stmts_) {
        if (!fullParts_.count(stmt)) {
            goto split_this_level;
        }
    }
    fullParts_.insert(op);
    return op;

split_this_level:
    for (auto &stmt : op->stmts_) {
        if (fullParts_.count(stmt)) {
            stmt = makeIf("", cond_, stmt);
        }
    }
    return op;
}

void MakeSync::markSyncForSplitting(const Stmt &sync) {
    bool inElseCase = false;
    for (auto ctx = cursor(); ctx.hasOuter(); ctx = ctx.outer()) {
        if (ctx.nodeType() == ASTNodeType::If) {
            (inElseCase ? branchSplittersElse_ : branchSplittersThen_)[ctx.id()]
                .emplace_back(sync);
        }
        inElseCase = ctx.outer().isValid() &&
                     ctx.outer().nodeType() == ASTNodeType::If &&
                     ctx.outer().node().as<IfNode>()->elseCase_ == ctx.node();
    }
}

Stmt MakeSync::visitStmt(const Stmt &op) {
    auto ret = BaseClass::visitStmt(op);
    // Please note that we have exited BaseClass, so `cursor()` is out
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
            markSyncForSplitting(sync);
        } else {
            syncBeforeFor_[whereToInsert.id()] = sync;
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
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    bool needSyncThreads = false, needSyncWarp = false;
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.lcaLoop_.node() == _op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
        }
    }
    if (needSyncThreads || needSyncWarp) {
        Stmt sync;
        if (needSyncThreads) {
            sync = makeEval(
                "", makeIntrinsic("__syncthreads()", {}, DataType::Void));
        } else if (needSyncWarp) {
            sync =
                makeEval("", makeIntrinsic("__syncwarp()", {}, DataType::Void));
        }
        op->body_ = makeStmtSeq("", {op->body_, sync});
        markSyncForSplitting(sync);
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
        auto &&sync = syncBeforeFor_.at(op->id());
        markSyncForSplitting(sync);
        return makeStmtSeq("", {sync, op});
    }
    return op;
}

Stmt MakeSync::visit(const If &_op) {
    auto __op = BaseClass::visit(_op);
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

        Stmt thenBody = op->thenCase_;
        if (branchSplittersThen_.count(op->id())) {
            CopyParts thenCopier(op->cond_, branchSplittersThen_.at(op->id()));
            thenBody = thenCopier(thenBody);
        }
        if (op->elseCase_.isValid()) {
            Stmt elseBody = op->elseCase_;
            if (branchSplittersElse_.count(op->id())) {
                CopyParts elseCopier(makeLNot(op->cond_),
                                     branchSplittersElse_.at(op->id()));
                elseBody = elseCopier(elseBody);
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

    return mergeAndHoistIf(op);
}

} // namespace gpu

} // namespace ir

