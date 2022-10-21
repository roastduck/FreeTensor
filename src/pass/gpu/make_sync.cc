#ifdef FT_WITH_CUDA

#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/const_fold.h>
#include <pass/gpu/make_sync.h>
#include <pass/merge_and_hoist_if.h>

namespace freetensor {

namespace gpu {

void FindAllThreads::visit(const For &op) {
    if (op->property_->parallel_ == threadIdxX) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            thx_ = op->len_.as<IntConstNode>()->val_;
        } else {
            thx_ = std::nullopt;
        }
    } else if (op->property_->parallel_ == threadIdxY) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            thy_ = op->len_.as<IntConstNode>()->val_;
        } else {
            thy_ = std::nullopt;
        }
    } else if (op->property_->parallel_ == threadIdxZ) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            thz_ = op->len_.as<IntConstNode>()->val_;
        } else {
            thz_ = std::nullopt;
        }
    }
    Visitor::visit(op);
    if (op->property_->parallel_ == threadIdxX) {
        results_[op->id()] =
            ThreadInfo{op, thx_.has_value() && warpSize_ % *thx_ == 0};
    } else if (op->property_->parallel_ == threadIdxY) {
        results_[op->id()] =
            ThreadInfo{op, thx_.has_value() && thy_.has_value() &&
                               warpSize_ % (*thx_ * *thy_) == 0};
    } else if (op->property_->parallel_ == threadIdxZ) {
        results_[op->id()] = ThreadInfo{
            op, thx_.has_value() && thy_.has_value() && thz_.has_value() &&
                    warpSize_ % (*thx_ * *thy_ * *thz_) == 0};
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
        if (op->property_->parallel_ == serialScope &&
            (op->begin_->nodeType() != ASTNodeType::IntConst ||
             op->end_->nodeType() != ASTNodeType::IntConst)) {
            throw InvalidProgram(
                "Unable to insert a synchronizing statment because it requires "
                "splitting a dynamic loop " +
                toString(op->id()) +
                " into two parts, to avoid synchronizing inside a condition "
                "of " +
                toString(cond_));
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
            stmt = makeIf(cond_, stmt);
        }
    }
    return op;
}

void MakeSync::markSyncForSplitting(const Stmt &stmtInTree, const Stmt &sync,
                                    bool needSyncWarp) {
    if (needSyncWarp) {
        return;
    }
    bool inElseCase = false;
    for (auto ctx = stmtInTree; ctx.isValid(); ctx = ctx->parentStmt()) {
        if (ctx->nodeType() == ASTNodeType::If) {
            auto branch = ctx.as<IfNode>();
            bool needSplitBranch = false;
            for (auto &&[loop, threadInfo] : loop2thread_) {
                if (isVariant(variantExprs_, {branch->cond_, branch}, loop)) {
                    needSplitBranch = true;
                    break;
                }
            }
            if (needSplitBranch) {
                (inElseCase ? branchSplittersElse_
                            : branchSplittersThen_)[ctx->id()]
                    .emplace_back(sync);
            }
        }
        inElseCase = ctx->parentStmt().isValid() &&
                     ctx->parentStmt()->nodeType() == ASTNodeType::If &&
                     ctx->parentStmt().as<IfNode>()->elseCase_ == ctx;
    }
}

Stmt MakeSync::visitStmt(const Stmt &op) {
    auto ret = BaseClass::visitStmt(op);

    Stmt target;
    bool needSyncThreads = false, needSyncWarp = false;
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.later_ == op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
            if (!target.isValid() || dep.lcaStmt_->depth() > target->depth()) {
                target = dep.lcaStmt_;
            }
        }
    }

    if (needSyncThreads || needSyncWarp) {
        Stmt sync;
        if (needSyncThreads) {
            sync = makeEval(
                makeIntrinsic("__syncthreads()", {}, DataType::Void, true));
        } else {
            sync = makeEval(
                makeIntrinsic("__syncwarp()", {}, DataType::Void, true));
        }

        Stmt whereToInsert;
        for (auto ctx = op->parentStmt(); ctx->id() != target->id();
             ctx = ctx->parentStmt()) {
            if (ctx->nodeType() == ASTNodeType::For) {
                whereToInsert = ctx;
            }
        }
        if (!whereToInsert.isValid()) {
            ret = makeStmtSeq({sync, ret});
            markSyncForSplitting(op->parentStmt(), sync, needSyncWarp);
        } else {
            syncBeforeFor_[whereToInsert->id()] = sync;
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
        if (!dep.synced_ && !dep.visiting_ && dep.earlier_ == op) {
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
        if (!dep.synced_ && dep.visiting_ && dep.lcaLoop_ == _op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
        }
    }
    if (needSyncThreads || needSyncWarp) {
        Stmt sync;
        if (needSyncThreads) {
            sync = makeEval(
                makeIntrinsic("__syncthreads()", {}, DataType::Void, true));
        } else if (needSyncWarp) {
            sync = makeEval(
                makeIntrinsic("__syncwarp()", {}, DataType::Void, true));
        }
        op->body_ = makeStmtSeq({op->body_, sync});
        markSyncForSplitting(_op, sync, needSyncWarp);
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
        markSyncForSplitting(_op, sync, needSyncWarp);
        return makeStmtSeq({sync, op});
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
            return makeStmtSeq({thenBody, elseBody});
        } else {
            return thenBody;
        }
    }
    return op;
}

Stmt makeSync(const Stmt &_op, const Ref<GPUTarget> &target) {
    auto op = constFold(_op);
    FindAllThreads finder(target);
    finder(op);
    auto &&loop2thread = finder.results();

    std::vector<FindDepsDir> query;
    query.reserve(loop2thread.size());
    for (auto &&[loopId, thr] : loop2thread) {
        query.push_back({{loopId, DepDirection::Different}});
    }
    std::vector<CrossThreadDep> deps;
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() == 1);
        auto commonStmt = lcaStmt(d.later_.stmt_, d.earlier_.stmt_);
        auto commonLoop = commonStmt;
        while (commonLoop->parent().isValid() &&
               commonLoop->nodeType() != ASTNodeType::For) {
            commonLoop = commonLoop->parentStmt();
        }
        ASSERT(commonLoop->nodeType() == ASTNodeType::For);
        deps.emplace_back(CrossThreadDep{
            d.later_.stmt_, d.earlier_.stmt_, commonStmt, commonLoop,
            loop2thread.at(d.dir_[0].first.id_).inWarp_, false, false});
    };
    FindDeps()
        .direction(query)
        .filterAccess([](const AccessPoint &acc) {
            return acc.buffer_->mtype() == MemType::GPUGlobal ||
                   acc.buffer_->mtype() == MemType::GPUShared;
        })
        .filter([](const AccessPoint &later, const AccessPoint &earlier) {
            return later.op_ != earlier.op_;
        })
        .ignoreReductionWAW(false)
        .eraseOutsideVarDef(false)(op, found);

    auto variantExprs = findLoopVariance(op).first;

    MakeSync mutator(op, loop2thread, std::move(deps), std::move(variantExprs));
    op = mutator(op);

    return mergeAndHoistIf(op);
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
