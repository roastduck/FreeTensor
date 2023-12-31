#ifdef FT_WITH_CUDA

#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds_combination.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <pass/const_fold.h>
#include <pass/gpu/make_sync.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/normalize_loops.h>

namespace freetensor {

namespace gpu {

namespace {

class RetryMakeSync : public Error {
    ID loopId_;

  public:
    RetryMakeSync(const ID &loopId, const std::string &msg,
                  std::source_location loc = std::source_location::current())
        : Error(msg, loc), loopId_(loopId) {}

    const auto &loopId() const { return loopId_; }
};

class RelaxOneLoop : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    ID loopId_;

  public:
    RelaxOneLoop(const ID &loopId) : loopId_(loopId) {}

  protected:
    using BaseClass::visit;

    Stmt visit(const For &_op) override {
        auto __op = BaseClass::visit(_op);
        ASSERT(_op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();

        if (op->id() == loopId_) {
            CompUniqueBoundsCombination bound(*this);
            // Already normalized
            ASSERT(op->begin_->nodeType() == ASTNodeType::IntConst &&
                   op->begin_.as<IntConstNode>()->val_ == 0);
            if (auto u = bound.getIntUpper(op->end_); u != LLONG_MAX) {
                op->body_ = makeIf(makeLT(makeVar(op->iter_), op->end_),
                                   std::move(op->body_));
                op->end_ = makeIntConst(u);
                op->len_ = makeIntConst(u);
            } else {
                throw InvalidProgram("Unable to relax " + toString(op->end_));
            }
        }
        return op;
    }
};

Stmt relaxOneLoop(const Stmt &_ast, const ID &loopId) {
    auto ast =
        normalizeLoops(_ast, [&](auto &&l) { return l->id() == loopId; });
    ast = RelaxOneLoop{loopId}(ast);
    return ast;
}

} // Anonymous namespace

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
    if (ret->children().empty()) { // leaf node
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
            // TODO: Dynamic lengths but evaluated to be the same value in each
            // blocks should also be accpetable here
            throw RetryMakeSync(
                op->id(), "Unable to insert a synchronizing statment because "
                          "it requires splitting a dynamic loop " +
                              toString(op->id()) +
                              " into two parts, to avoid synchronizing inside "
                              "a condition of " +
                              toString(cond_) + ".");
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

Stmt MakeSync::makeSyncThreads() {
    return makeEval(makeIntrinsic("__syncthreads()", {}, DataType::Void, true));
}

Stmt MakeSync::makeSyncWarp() {
    return makeEval(
        makeIntrinsic("__syncwarp(__activemask())", {}, DataType::Void, true));
}

void MakeSync::markSyncForSplitting(const Stmt &stmtInTree, const Stmt &sync,
                                    bool isSyncWarp) {
    if (isSyncWarp) {
        return;
    }
    for (Stmt ctx = stmtInTree->parentStmt(), childOfCtx = stmtInTree;
         ctx.isValid(); childOfCtx = ctx, ctx = ctx->parentStmt()) {
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
                if (childOfCtx == branch->thenCase_) {
                    branchSplittersThen_[ctx->id()].emplace_back(sync);
                } else {
                    ASSERT(childOfCtx == branch->elseCase_);
                    branchSplittersElse_[ctx->id()].emplace_back(sync);
                }
            }
        }
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
        Stmt sync = needSyncThreads ? makeSyncThreads() : makeSyncWarp();

        Stmt whereToInsert;
        for (auto ctx = op->parentStmt(); ctx->id() != target->id();
             ctx = ctx->parentStmt()) {
            if (ctx->nodeType() == ASTNodeType::For) {
                whereToInsert = ctx;
            } else if (ctx->nodeType() == ASTNodeType::If &&
                       ctx.as<IfNode>()->elseCase_.isValid()) {
                // Need to sync before an `if` only when there is an `else`
                // case, where we need the `then` case AND the `else` case to
                // sync on ONE sync point
                whereToInsert = ctx;
            }
        }
        if (!whereToInsert.isValid()) {
            ret = makeStmtSeq({sync, ret});
            markSyncForSplitting(op, sync, !needSyncThreads);
        } else if (whereToInsert->nodeType() == ASTNodeType::For) {
            if (!syncBeforeFor_.count(whereToInsert->id()) ||
                (needSyncThreads &&
                 syncBeforeFor_.at(whereToInsert->id()).second)) {
                syncBeforeFor_[whereToInsert->id()] = {sync, !needSyncThreads};
            }
        } else {
            ASSERT(whereToInsert->nodeType() == ASTNodeType::If);
            if (!syncBeforeIf_.count(whereToInsert->id()) ||
                (needSyncThreads &&
                 syncBeforeIf_.at(whereToInsert->id()).second)) {
                syncBeforeIf_[whereToInsert->id()] = {sync, !needSyncThreads};
            }
        }

        for (CrossThreadDep &dep : deps_) {
            if (dep.visiting_) {
                if (needSyncThreads) {
                    dep.synced_ = true;
                }
                if (needSyncWarp && dep.inWarp_) {
                    dep.synced_ = true;
                    if (!needSyncThreads) {
                        dep.syncedOnlyInBranch_ = true;
                    }
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
        Stmt sync = needSyncThreads ? makeSyncThreads() : makeSyncWarp();
        op->body_ = makeStmtSeq({op->body_, sync});
        markSyncForSplitting(_op->body_, sync, !needSyncThreads);
        for (CrossThreadDep &dep : deps_) {
            if (dep.visiting_) {
                if (needSyncThreads) {
                    dep.synced_ = true;
                }
                if (needSyncWarp && dep.inWarp_) {
                    dep.synced_ = true;
                    if (!needSyncThreads) {
                        dep.syncedOnlyInBranch_ = true;
                    }
                }
            }
        }
    }
    if (syncBeforeFor_.count(op->id())) {
        auto &&[sync, isSyncWarp] = syncBeforeFor_.at(op->id());
        markSyncForSplitting(_op, sync, isSyncWarp);
        return makeStmtSeq({sync, op});
    }
    return op;
}

Stmt MakeSync::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto thenCase = (*this)(op->thenCase_);
    for (auto &&dep : deps_) {
        if (dep.syncedOnlyInBranch_) {
            dep.synced_ = false;
        }
    }
    Stmt elseCase;
    if (op->elseCase_.isValid()) {
        elseCase = (*this)(op->elseCase_);
        for (auto &&dep : deps_) {
            if (dep.syncedOnlyInBranch_) {
                dep.synced_ = false;
            }
        }
    }

    Stmt ret;
    if (branchSplittersThen_.count(op->id()) ||
        branchSplittersElse_.count(op->id())) {
        if (!checkNotModified(root_, cond, CheckNotModifiedSide::Before,
                              op->id(), CheckNotModifiedSide::After,
                              op->id())) {
            throw InvalidProgram("Unable to insert a synchronizing statment "
                                 "inside an If node because the condition " +
                                 toString(cond) + " is being modified");
        }

        Stmt thenBody;
        if (branchSplittersThen_.count(op->id())) {
            CopyParts thenCopier(cond, branchSplittersThen_.at(op->id()));
            thenBody = thenCopier(thenCase);
        } else {
            thenBody = makeIf(cond, thenCase);
        }
        if (elseCase.isValid()) {
            Stmt elseBody;
            if (branchSplittersElse_.count(op->id())) {
                CopyParts elseCopier(makeLNot(cond),
                                     branchSplittersElse_.at(op->id()));
                elseBody = elseCopier(elseCase);
            } else {
                elseBody = makeIf(makeLNot(cond), elseCase);
            }
            ret = makeStmtSeq({thenBody, elseBody});
        } else {
            ret = thenBody;
        }
    } else {
        ret = makeIf(std::move(cond), std::move(thenCase), std::move(elseCase),
                     op->metadata(), op->id(), op->debugBlame());
    }

    if (syncBeforeIf_.count(op->id())) {
        auto &&[sync, isSyncWarp] = syncBeforeIf_.at(op->id());
        markSyncForSplitting(op, sync, isSyncWarp);
        return makeStmtSeq({sync, ret});
    }

    return ret;
}

static Stmt doMakeSync(const Stmt &_op, const Ref<GPUTarget> &target) {
    auto op = constFold(_op);

    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        if (!lcaStmt(later.stmt_, earlier.stmt_)
                 ->parentStmtByFilter([&](const Stmt &s) {
                     return s->nodeType() == ASTNodeType::For &&
                            std::holds_alternative<CUDAScope>(
                                s.as<ForNode>()->property_->parallel_);
                 })
                 .isValid()) {
            // Crossing kernels, skipping
            return false;
        }

        // Already synchronized for specific `ReduceTo` node (for example by
        // using atomic). No need for additional `__syncthreads`.
        //
        // NOTE 1: We prefer `__syncthreads` over synchronizing individual
        // `ReduceTo` nodes: We check in `pass/make_parallel_reduction`, if
        // we are able to use `__syncthreads` here, we won't set `sync_`
        // there.
        //
        // NOTE 2: Our schedules prohibits simultaneous `Load` and
        // `ReduceTo`, or simultaneous `Store` and `ReduceTo`, so
        // synchronizations for `ReduceTo` nodes may only synchronize among
        // `ReduceTo` nodes, not between `ReduceTo` and `Store`, or between
        // `ReduceTo` and `Load`. But after `pass/normalize_threads`,
        // everything can happen simultaneously (and that's why we need this
        // `pass/make_sync`). Thus we can only ignore the synchronization
        // only if BOTH `earlier` and `later` are `ReduceTo`.
        if (later.op_->nodeType() == ASTNodeType::ReduceTo &&
            earlier.op_->nodeType() == ASTNodeType::ReduceTo &&
            later.op_.as<ReduceToNode>()->sync_ &&
            earlier.op_.as<ReduceToNode>()->sync_) {
            return false;
        }

        return later.op_ != earlier.op_;
    };

    // Reject all cross-block dependences
    std::vector<FindDepsDir> queryBlocks;
    for (auto &&loop : findAllStmt(op, [](const Stmt &s) {
             if (s->nodeType() == ASTNodeType::For) {
                 auto &&p = s.as<ForNode>()->property_->parallel_;
                 return std::holds_alternative<CUDAScope>(p) &&
                        std::get<CUDAScope>(p).level_ == CUDAScope::Block;
             }
             return false;
         })) {

        // The block ID is different
        FindDepsDir queryOneBlock = {{loop->id(), DepDirection::Different}};

        // But in the same kernel invocation
        for (auto p = loop->parentStmt(); p.isValid(); p = p->parentStmt()) {
            if (p->nodeType() == ASTNodeType::For &&
                p.as<ForNode>()->property_->parallel_ == serialScope) {
                queryOneBlock.push_back({p->id(), DepDirection::Same});
            }
        }

        queryBlocks.emplace_back(std::move(queryOneBlock));
    }
    FindDeps()
        .direction(queryBlocks)
        .filterAccess([](const auto &acc) {
            return acc.buffer_->mtype() == MemType::GPUGlobal ||
                   acc.buffer_->mtype() == MemType::GPUGlobalHeap;
        })
        .filter(filter)
        .ignoreReductionWAW(false)
        .eraseOutsideVarDef(false)(op, [](const Dependence &d) {
            throw InvalidProgram("Dependence between blocks in a single CUDA "
                                 "kernel cannot be resolved: " +
                                 toString(d));
        });

    FindAllThreads finder(target);
    finder(op);
    auto &&loop2thread = finder.results();

    std::vector<FindDepsDir> query;
    query.reserve(loop2thread.size());
    for (auto &&[loopId, thr] : loop2thread) {
        query.push_back({{loopId, DepDirection::Different}});
    }
    std::vector<CrossThreadDep> deps;
    auto found = [&](const Dependence &d) {
        ASSERT(d.dir_.size() == 1);
        auto laterStmt = d.later_.stmt_;
        auto earlierStmt = d.earlier_.stmt_;
        auto commonStmt = lcaStmt(laterStmt, earlierStmt);
        if (commonStmt == laterStmt || commonStmt == earlierStmt) {
            // Dependence between a If's condition or a For's range and its
            // body. We can treat it as the If or For node depending on itself.
            // E.g.,
            //
            // ```
            // for ... {
            //   if x { // THIS IF DEPENDS ON ITSELF
            //     x = 1
            //   }
            // }
            // ```
            laterStmt = earlierStmt = commonStmt;
        }
        auto commonLoop = commonStmt;
        while (commonLoop->parent().isValid() &&
               commonLoop->nodeType() != ASTNodeType::For) {
            commonLoop = commonLoop->parentStmt();
        }
        ASSERT(commonLoop->nodeType() == ASTNodeType::For);
        deps.emplace_back(laterStmt, earlierStmt, commonStmt, commonLoop,
                          loop2thread.at(d.dir_[0].first.id_).inWarp_);
    };
    FindDeps()
        .direction(query)
        .filterAccess([](const auto &acc) {
            return acc.buffer_->mtype() == MemType::GPUGlobal ||
                   acc.buffer_->mtype() == MemType::GPUGlobalHeap ||
                   acc.buffer_->mtype() == MemType::GPUShared;
        })
        .filter(filter)
        .ignoreReductionWAW(false)
        .eraseOutsideVarDef(false)(op, found);

    auto variantExprs = findLoopVariance(op).first;

    MakeSync mutator(op, loop2thread, std::move(deps), std::move(variantExprs));
    op = mutator(op);

    return mergeAndHoistIf(op);
}

Stmt makeSync(const Stmt &_op, const Ref<GPUTarget> &target) {
    auto op = _op;
    while (true) {
        try {
            return doMakeSync(op, target);
        } catch (const RetryMakeSync &e1) {
            try {
                op = relaxOneLoop(op, e1.loopId());
            } catch (const InvalidProgram &e2) {
                throw InvalidProgram(
                    std::string(e1.what()) +
                    " Tried to relax the loop, but: " + e2.what());
            }
        }
    }
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
