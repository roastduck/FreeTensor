#include <algorithm>

#include <itertools.hpp>

#include <analyze/all_defs.h>
#include <analyze/all_stmts.h>
#include <analyze/count_contig_access_loops.h>
#include <analyze/find_indexing_loops.h>
#include <analyze/find_stmt.h>
#include <analyze/get_loop_nest_tree.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen.h>
#include <container_utils.h>
#include <driver.h>
#include <lower.h>
#include <omp_utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/simplify.h>
#include <schedule.h>
#include <schedule/as_matmul.h>
#include <schedule/blend.h>
#include <schedule/cache.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
#include <schedule/fuse.h>
#include <schedule/inlining.h>
#include <schedule/merge.h>
#include <schedule/multi_level_tiling.h>
#include <schedule/parallelize.h>
#include <schedule/reorder.h>
#include <schedule/separate_tail.h>
#include <schedule/set_mem_type.h>
#include <schedule/split.h>
#include <schedule/swap.h>
#include <schedule/unroll.h>
#include <schedule/var_merge.h>
#include <schedule/var_reorder.h>
#include <schedule/vectorize.h>

namespace freetensor {

Schedule::Schedule(const Stmt &ast, int verbose)
    : ast_(ast), verbose_(verbose), memoized_(Ref<MemoizedSchedules>::make()),
      rng_(Ref<OpenMPRandomEngine>::make(0)) /* TODO: set seed */,
      randCtx_(Ref<RandCtx<OpenMPRandomEngine>>::make(*rng_)) {
    ast_ = simplify(ast_);
}

void Schedule::saveSuccessLog(const ScheduleLog &logs) {
    logs_ = logs;
    if (verbose_ >= 2) {
        logger() << "AST after " << *logs.top() << " is:" << std::endl
                 << ast_ << std::endl;
    }
}

Stmt Schedule::ast() const {
    if (verbose_ >= 1) {
        logger() << "The scheduled AST is:" << std::endl << ast_ << std::endl;
    }
    return ast_;
}

std::vector<Ref<ScheduleLogItem>> Schedule::logs() const {
    std::vector<Ref<ScheduleLogItem>> revRet;
    for (auto it = logs_; !it.empty(); it = it.pop()) {
        revRet.emplace_back(it.top());
    }
    return std::vector<Ref<ScheduleLogItem>>(revRet.rbegin(), revRet.rend());
}

std::vector<Stmt>
Schedule::findAll(const std::function<bool(const Stmt &)> &filter) const {
    return findStmt(ast_, filter);
}

Stmt Schedule::find(const std::function<bool(const Stmt &)> &filter) const {
    auto ret = findStmt(ast_, filter);
    if (ret.size() != 1) {
        throw InvalidSchedule("find: There is " + std::to_string(ret.size()) +
                              " nodes matching the given condition. "
                              "Consider using findAll");
    }
    return ret[0];
}

// Make a log item with specifc parameter and result types
#define MAKE_LOG(TYPE, FUNC, ...)                                              \
    ([&](const auto &func, const auto &params) {                               \
        /* decay is required: we must not store an reference */                \
        typedef ScheduleLogItemImpl<                                           \
            ScheduleType::TYPE, std::decay_t<decltype(func)>,                  \
            std::decay_t<decltype(params)>,                                    \
            std::decay_t<decltype(std::apply(func, params))>>                  \
            BaseClass;                                                         \
        class ScheduleLogItem##TYPE : public BaseClass {                       \
          public:                                                              \
            ScheduleLogItem##TYPE(const typename BaseClass::Invocable &f,      \
                                  const typename BaseClass::Params &p)         \
                : BaseClass(f, p) {}                                           \
        };                                                                     \
        return Ref<ScheduleLogItem##TYPE>::make(func, params);                 \
    })(FUNC, std::make_tuple(__VA_ARGS__))

// - Try looking up an identical schedule from `MemoizedSchedules`
// - If not found, do the schedule (if found, `run` directly returns)
// - Save the result (including exceptions, if any) back to `MemoziedSchedules`
#define RUN_SCHEDULE_MEMORIZEDLY(logs, log)                                    \
    logs = memoized_->lookup(logs);                                            \
    ASSERT(logs.top()->type() == log->type());                                 \
    log = logs.top().as<decltype(log)::Object>();                              \
    log->run();                                                                \
    memoized_->save(logs);

std::pair<ID, ID> Schedule::split(const ID &id, int factor, int nparts,
                                  int shift) {
    auto log = MAKE_LOG(Split, std::bind_front(freetensor::split, ast_), id,
                        factor, nparts, shift);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::reorder(const std::vector<ID> &order) {
    auto log =
        MAKE_LOG(Reorder, std::bind_front(freetensor::reorder, ast_), order);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

ID Schedule::merge(const ID &loop1, const ID &loop2) {
    auto log =
        MAKE_LOG(Merge, std::bind_front(freetensor::merge, ast_), loop1, loop2);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const ID &loop, FissionSide side, const ID &splitter,
                  const std::string &suffix0, const std::string &suffix1) {
    auto log = MAKE_LOG(Fission, std::bind_front(freetensor::fission, ast_),
                        loop, side, splitter, suffix0, suffix1);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

ID Schedule::fuse(const ID &loop0, const ID &loop1, bool strict) {
    auto log = MAKE_LOG(Fuse, std::bind_front(freetensor::fuse, ast_), loop0,
                        loop1, strict);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

ID Schedule::fuse(const ID &loop0, bool strict) {
    ast_ = flattenStmtSeq(ast_);
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
                return fuse(loop0, s->id(), strict);
            }
        }
    }
    throw InvalidSchedule("Invalid fuse(" + toString(loop0) +
                          "): Unable to find a following loop of " +
                          toString(loop0));
}

void Schedule::swap(const std::vector<ID> &order) {
    auto log = MAKE_LOG(Swap, std::bind_front(freetensor::swap, ast_), order);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::blend(const ID &loop) {
    auto log = MAKE_LOG(Blend, std::bind_front(freetensor::blend, ast_), loop);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cache(const ID &stmt, const std::string &var, MemType mtype) {
    auto log = MAKE_LOG(Cache, std::bind_front(freetensor::cache, ast_), stmt,
                        var, mtype);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cacheReduction(const ID &stmt, const std::string &var,
                         MemType mtype) {
    auto log = MAKE_LOG(CacheReduction,
                        std::bind_front(freetensor::cacheReduction, ast_), stmt,
                        var, mtype);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        auto ret = log->getResult();
        ast_ = ret.first;
        saveSuccessLog(logs);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::setMemType(const ID &def, MemType mtype) {
    auto log = MAKE_LOG(
        SetMemType, std::bind_front(freetensor::setMemType, ast_), def, mtype);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::varSplit(const ID &def, int dim, VarSplitMode mode, int factor,
                        int nparts) {
    auto log = MAKE_LOG(VarSplit, std::bind_front(freetensor::varSplit, ast_),
                        def, dim, mode, factor, nparts);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::varMerge(const ID &def, int dim) {
    auto log = MAKE_LOG(VarMerge, std::bind_front(freetensor::varMerge, ast_),
                        def, dim);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::varReorder(const ID &def, const std::vector<int> &order) {
    auto log = MAKE_LOG(
        VarReorder, std::bind_front(freetensor::varReorder, ast_), def, order);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

ID Schedule::moveTo(const ID &_stmt, MoveToSide side, const ID &_dst) {
    auto bak = ast_;
    try {
        auto stmt = _stmt, dst = _dst;
        while (true) {
            ast_ = hoistVarOverStmtSeq(ast_);
            Stmt s = findStmt(ast_, stmt);
            Stmt d = findStmt(ast_, dst);

            auto movingUp = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::Before;
                }
                if (auto prev = s->prevStmt(); prev.isValid()) {
                    return d->isBefore(side == MoveToSide::After ? prev : s);
                } else {
                    return d->isBefore(s);
                }
            };
            auto movingDown = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::After;
                }
                if (auto next = s->nextStmt(); next.isValid()) {
                    return (side == MoveToSide::Before ? next : s)->isBefore(d);
                } else {
                    return s->isBefore(d);
                }
            };

            if (movingUp()) {
                if (s->prevStmt().isValid()) {
                    std::vector<ID> orderRev;
                    while (s->prevStmt().isValid() && movingUp()) {
                        s = s->prevStmt();
                        orderRev.emplace_back(s->id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<ID> order(orderRev.rbegin(), orderRev.rend());
                    swap(order);
                } else {
                    while (!s->prevStmt().isValid() && movingUp()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast_,
                            "Fission a If node in a StmtSeq is not currently "
                            "supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMap =
                        fission(s->id(), FissionSide::After, stmt, ".a", "")
                            .first;
                    stmt = idMap.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else if (movingDown()) {
                if (s->nextStmt().isValid()) {
                    std::vector<ID> order;
                    while (s->nextStmt().isValid() && movingDown()) {
                        s = s->nextStmt();
                        order.emplace_back(s->id());
                    }
                    order.emplace_back(stmt);
                    swap(order);
                } else {
                    while (!s->nextStmt().isValid() && movingDown()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast_,
                            "Fission a If node in a StmtSeq is not currently "
                            "supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMap =
                        fission(s->id(), FissionSide::Before, stmt, "", ".b")
                            .second;
                    stmt = idMap.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else {
                return s->id();
            }
        }
    } catch (const InvalidSchedule &e) {
        ast_ = bak;
        throw InvalidSchedule(ast_, "Invalid move_to(" + toString(_stmt) +
                                        ", " + toString(_dst) +
                                        "): " + e.what());
    }
}

void Schedule::inlining(const ID &def) {
    auto log =
        MAKE_LOG(Inline, std::bind_front(freetensor::inlining, ast_), def);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::parallelize(const ID &loop, const ParallelScope &parallel) {
    auto log =
        MAKE_LOG(Parallelize, std::bind_front(freetensor::parallelize, ast_),
                 loop, parallel);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::unroll(const ID &loop, bool immediate) {
    auto log = MAKE_LOG(Unroll, std::bind_front(freetensor::unroll, ast_), loop,
                        immediate);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::vectorize(const ID &loop) {
    auto log =
        MAKE_LOG(Vectorize, std::bind_front(freetensor::vectorize, ast_), loop);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::separateTail(bool noDuplicateVarDefs) {
    auto log =
        MAKE_LOG(SeparateTail, std::bind_front(freetensor::separateTail, ast_),
                 noDuplicateVarDefs);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::asMatMul(const ID &loop) {
    auto log =
        MAKE_LOG(AsMatMul, std::bind_front(freetensor::asMatMul, ast_), loop);
    ScheduleLog logs = logs_.push(log);
    RUN_SCHEDULE_MEMORIZEDLY(logs, log);
    try {
        ast_ = log->getResult();
        saveSuccessLog(logs);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log, ast_, e.what());
    }
}

void Schedule::autoSchedule(const Target &target, const Ref<RandTrace> &trace) {
    autoUseLib(target);
    autoFuse(target, trace);
    autoParallelize(target);
    autoSetMemType(target);
    autoUnroll(target);
}

void Schedule::autoUseLib(const Target &target) {
    // Try to implement each top-level loops with lib calls
    auto loopNestTree = getLoopNestTree(ast_);
    for (auto &&loop : loopNestTree->subLoops_) {
        try {
            asMatMul(loop->loop_->id());
        } catch (const InvalidSchedule &e) {
            // If the loop is marked as preferLibs, we inline all local
            // variables, fission all the statments apart, and try applying to
            // each of them
            bool isPreferLibs = false;
            for (For l = loop->loop_;;) {
                if (l->property_->preferLibs_) {
                    isPreferLibs = true;
                    break;
                }
                Stmt body = l->body_;
                while (body->nodeType() == ASTNodeType::VarDef) {
                    body = body.as<VarDefNode>()->body_;
                }
                if (body->nodeType() != ASTNodeType::For) {
                    break;
                } else {
                    l = body.as<ForNode>();
                }
            }
            if (isPreferLibs) {
                for (auto &&[defId, name] :
                     allDefs(loop->loop_, {AccessType::Cache})) {
                    try {
                        inlining(defId);
                    } catch (const InvalidSchedule &e) {
                        // do nothing
                    }
                }
                auto stmts = allStmts(
                    loop->loop_, {ASTNodeType::Store, ASTNodeType::ReduceTo});
                for (auto &&[i, stmt] : iter::enumerate(stmts)) {
                    auto bak = ast_;
                    auto logBak = logs_;
                    try {
                        fission(loop->loop_->id(), FissionSide::Before,
                                stmt->id(), "." + std::to_string(i), "");
                        auto libStmtId =
                            fission(loop->loop_->id(), FissionSide::After,
                                    stmt->id(),
                                    "." + std::to_string(i) + ".lib", "")
                                .first.at(loop->loop_->id());
                        asMatMul(libStmtId);
                    } catch (const InvalidSchedule &e) {
                        ast_ = std::move(bak), logs_ = std::move(logBak);
                    }
                }
            }
        }
    }
}

void Schedule::autoFuse(const Target &target, const Ref<RandTrace> &trace) {
    // Try to fuse each pair of consecutive loops
    std::function<void(const Ref<LoopNest> &nest)> visitNest =
        [&, this](const Ref<LoopNest> &nest) {
            Ref<LoopNest> last;
            ID lastId;
            for (auto &&subNest : nest->subLoops_) {
                auto thisId = subNest->loop_->id();
                if (last.isValid() &&
                    randCtx_->decide(PROGRAM_POSITION, "fuse", {0, 1}, trace)) {
                    auto bak = ast_;
                    auto logBak = logs_;
                    try {
                        try {
                            lastId = moveTo(lastId, MoveToSide::Before, thisId);
                        } catch (const InvalidSchedule &e) {
                            thisId = moveTo(thisId, MoveToSide::After, lastId);
                        }
                        thisId = fuse(lastId, thisId, true);
                        subNest->subLoops_.insert(subNest->subLoops_.begin(),
                                                  last->subLoops_.begin(),
                                                  last->subLoops_.end());
                        last->subLoops_.clear();
                    } catch (const InvalidSchedule &e) {
                        ast_ = std::move(bak), logs_ = std::move(logBak);
                        visitNest(last);
                    }
                }
                lastId = thisId, last = subNest;
            }
            if (last.isValid()) {
                visitNest(last);
            }
        };
    visitNest(getLoopNestTree(ast_));
}

void Schedule::autoParallelize(const Target &target) {
    // [GPU only] Try to parallelize loops accessing contiguous items as warps
    if (target.type() == TargetType::GPU) {
        // We try to parallelize the loop with most contiguous access count
        // first. If the counts are equal, we try to parallel the out-most loop
        // with the same count first
        CountContigAccessLoops contigFinder;
        contigFinder(ast_);
        std::vector<std::pair<ID, std::pair<int64_t, int>>> contigLoops(
            contigFinder.counts().begin(), contigFinder.counts().end());
        std::sort(contigLoops.begin(), contigLoops.end(),
                  [](const std::pair<ID, std::pair<int64_t, int>> &lhs,
                     const std::pair<ID, std::pair<int64_t, int>> &rhs) {
                      return lhs.second > rhs.second;
                  });
        for (auto &&[loopId, cnt] : contigLoops) {
            auto loop = find(loopId);

            // Ignore if too short
            if (auto &&len = loop.as<ForNode>()->len_;
                len->nodeType() == ASTNodeType::IntConst &&
                len.as<IntConstNode>()->val_ <= 4) {
                continue;
            }

            auto bak = ast_;
            auto logBak = logs_;
            try {
                // FIXME: Do not hard-code 32
                auto [l0, l1] = split(loop->id(), 32);
                parallelize(l1, threadIdxX);

                try {
                    // Reorder this scope to as outer as possible
                    auto refCntHolder = ast_;
                    auto c = find(l1);
                    if (c->parentStmt().isValid()) {
                        for (c = c->parentStmt(); c->parentStmt().isValid();
                             c = c->parentStmt()) {
                            if (c->nodeType() == ASTNodeType::For) {
                                try {
                                    reorder({l1, c->id()});
                                } catch (InvalidSchedule &e) {
                                    break;
                                }
                            }
                        }
                    }
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
            } catch (const InvalidSchedule &e) {
                ast_ = std::move(bak), logs_ = std::move(logBak);
            }
        }
    }

    // Try to merge and parallelize as many outer loops as possible
    std::function<void(const Ref<LoopNest> &)> autoParallelizeOuter =
        [&](const Ref<LoopNest> &root) {
            auto latestSuccess = ast_;
            auto successLogs = logs_;

            bool atLeastOne = false; // if at least one loop is parallelized
            try {
                Ref<LoopNest> loop = root;

                bool parentIsWarp = false;
                while (loop->loop_->property_->parallel_ != serialScope &&
                       loop->subLoops_.size() == 1) {
                    loop = loop->subLoops_.front();
                    parentIsWarp = true;
                }

                ID loopId, outerId;
                while (true) {
                    loopId = loop->loop_->id();
                    if (find(loopId).as<ForNode>()->property_->parallel_ !=
                        serialScope) {
                        break;
                    }
                    if (outerId.isValid()) {
                        loopId = merge(outerId, loopId);
                    }

                    auto bak = ast_;
                    auto logBak = logs_;
                    switch (target.type()) {
                    case TargetType::CPU:
                        parallelize(loopId, OpenMPScope{});
                        atLeastOne = true;
                        break;

                    case TargetType::GPU: {
                        auto loop = find(loopId);
                        auto isParallelLoop = [](const Stmt &s) {
                            return s->nodeType() == ASTNodeType::For &&
                                   s.as<ForNode>()->property_->parallel_ !=
                                       serialScope;
                        };
                        bool childIsWarp =
                            !findStmt(loop, isParallelLoop).empty();
                        // We guarantee the following requirements in order:
                        // 1. make sure all SMs are used
                        // 2. if there are enough threads, make sure blockDim is
                        // not too large If the loop length is constant, we
                        // split it only once, to reduce redundant guards, and
                        // save time for dependency analysis. If not, we split
                        // it twice, and merge once
                        int numSM = 80;
                        int maxThreads =
                            (!parentIsWarp && !childIsWarp) ? 256 : 8;
                        // TODO: do not hard-code these numbers
                        ID l1, l1b, l2;
                        if (auto loopNode = loop.as<ForNode>();
                            loopNode->len_->nodeType() ==
                            ASTNodeType::IntConst) {
                            auto len = loopNode->len_.as<IntConstNode>()->val_;
                            if (len < numSM * maxThreads) {
                                std::tie(l1, l2) = split(loopId, -1, numSM);
                            } else {
                                std::tie(l1, l2) = split(loopId, maxThreads);
                            }
                        } else {
                            // We don't use the `nparts` mode of `split`,
                            // because it will hinder dependency analysis.
                            // Instead, we use the `factor` mode and then
                            // reorder. See the doc string of `split` for
                            // details
                            std::tie(l2, l1) = split(loopId, numSM);
                            reorder({l1, l2});
                            if (!findAll(l2).empty()) {
                                std::tie(l1b, l2) = split(l2, maxThreads);
                            }
                        }
                        if (!findAll(l1).empty()) {
                            if (l1b.isValid() && !findAll(l1b).empty()) {
                                // We are unable to fuse `l1` and `l1b` back to
                                // one loop. Because the length of `l1b` is not
                                // a constant, a division by this length will be
                                // introduced, which is not supported by ISL and
                                // may probably lead to false dependencies
                                parallelize(l1, blockIdxY);
                                atLeastOne = true;
                                parallelize(l1b, blockIdxX);
                            } else {
                                parallelize(l1, blockIdxX);
                                atLeastOne = true;
                            }
                        }
                        if (!findAll(l2).empty()) {
                            parallelize(l2, (!parentIsWarp && !childIsWarp)
                                                ? threadIdxX
                                                : threadIdxY);
                            atLeastOne = true;
                        }
                        break;
                    }
                    default:
                        ASSERT(false);
                    }
                    latestSuccess = ast_, successLogs = logs_;
                    ast_ = std::move(bak), logs_ = std::move(logBak);

                    if (loop->subLoops_.size() == 1) {
                        outerId = loopId;
                        loop = loop->subLoops_.front();
                    } else {
                        break;
                    }
                }
            } catch (InvalidSchedule &e) {
                // do nothing
            }

            ast_ = latestSuccess, logs_ = successLogs;

            if (!atLeastOne) {
                for (auto &&subLoop : root->subLoops_) {
                    autoParallelizeOuter(subLoop);
                }
            }
        };
    auto loopNestTree = getLoopNestTree(ast_);
    for (const Ref<LoopNest> &root : loopNestTree->subLoops_) {
        // If the outer most loop is too short, we try the second outer loops
        // instead
        if (root->subLoops_.size() > 1 &&
            root->loop_->len_->nodeType() == ASTNodeType::IntConst &&
            root->loop_->len_.as<IntConstNode>()->val_ < 32) {
            for (const Ref<LoopNest> &root2 : root->subLoops_) {
                autoParallelizeOuter(root2);
            }
        } else {
            autoParallelizeOuter(root);
        }
    }
}

void Schedule::autoSetMemType(const Target &target) {
    // Try to put each VarDef as near to processor as possible
    if (target.type() == TargetType::GPU) {
        for (auto &&[defId, name] : allDefs(ast_, {AccessType::Cache})) {
            try {
                setMemType(defId, MemType::GPULocal);
            } catch (const InvalidSchedule &e) {
                try {
                    setMemType(defId, MemType::GPUShared);
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
            }
        }
    }
}

void Schedule::autoUnroll(const Target &target) {
    if (target.type() == TargetType::GPU) {
        // Try to unroll loops that accessing local arrays, to help nvcc put
        // these arrays to registers
        for (auto &&[loop, defs] : findIndexingLoops(ast_)) {
            if (loop->property_->parallel_ != serialScope ||
                loop->property_->vectorize_) {
                continue;
            }

            for (auto &&def : defs) {
                if (def->buffer_->mtype() == MemType::GPULocal) {
                    goto do_unroll;
                }
            }
            continue;
        do_unroll:
            try {
                unroll(loop->id());
            } catch (InvalidSchedule &e) {
                // do nothing
            }
        }
    }

    // Unroll very short loops
    std::function<void(const Ref<LoopNest> &nest)> visitNest =
        [&, this](const Ref<LoopNest> &nest) {
            auto &&loop = nest->loop_;
            if (loop.isValid()) { // not root
                if (loop->property_->parallel_ == serialScope &&
                    !loop->property_->vectorize_ && !loop->property_->unroll_ &&
                    loop->len_->nodeType() == ASTNodeType::IntConst &&
                    loop->len_.as<IntConstNode>()->val_ <= 4) {
                    unroll(loop->id());
                }
            }
            for (auto &&subNest : nest->subLoops_) {
                visitNest(subNest);
            }
        };
    visitNest(getLoopNestTree(ast_));
}

std::vector<AutoScheduleTuneTrial> Schedule::tuneAutoSchedule(
    int nBatch, int batchSize, const Ref<Device> &device,
    const std::vector<Ref<Array>> &args,
    const std::unordered_map<std::string, Ref<Array>> &kws,
    const std::regex &toLearn) {
    try {
        randCtx_->setLearn();
        randCtx_->setLearnFilter(toLearn);
        std::vector<AutoScheduleTuneTrial> trials(nBatch * batchSize);
        for (int i = 0; i < nBatch; i++) {
            if (verbose_ >= 1) {
                logger() << "Tuning auto_schedule: Batch " << i << std::endl;
            }
            exceptSafeParallelFor<size_t>(
                0, batchSize, 1,
                [&](size_t j) {
                    auto &[trace, lowered, code, _1, _2] =
                        trials[i * batchSize + j];
                    auto s = this->fork();
                    trace = Ref<RandTrace>::make();
                    s.autoSchedule(*device->target(), trace);
                    lowered = lower(s.func(), device->target());
                    code = codeGen(lowered, device->target());
                },
                omp_sched_static); // use schedule(static) to guarantee
                                   // deterministic RNG
            for (int j = 0; j < batchSize; j++) {
                auto &[trace, lowered, code, t, stddev] =
                    trials[i * batchSize + j];
                Driver d(lowered, code, device);
                d.setArgs(args, kws);
                // TODO: Allow setting measuring repeats
                std::tie(t, stddev) = d.time();
                d.collectReturns();
                randCtx_->observeTrace(trace, t, stddev);
            }
        }
        randCtx_->setInfer();
        return trials;
    } catch (...) {
        randCtx_->setInfer();
        return {};
    }
}

std::vector<std::pair<ID, int>>
Schedule::multiLevelTiling(const ForsWithDataReuse &target,
                           const MultiLevelTilingAnnotation &annotation,
                           const std::string &pat, int level) {
    return freetensor::multiLevelTiling(*this, target, annotation, pat, level);
}
std::vector<std::pair<ID, int>> Schedule::multiLevelTilingWithFusion(
    const ForsWithDataReuse &target,
    const MultiLevelTilingAnnotation &annotation, const std::string &pat,
    const ElementWiseInfo &toFuse, int level, TargetType targetType,
    bool doCacheRead) {
    return freetensor::multiLevelTilingWithFusion(
        *this, target, annotation, pat, toFuse, level, targetType, doCacheRead);
}
} // namespace freetensor
