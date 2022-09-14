#include <algorithm>

#include <analyze/all_defs.h>
#include <analyze/all_stmts.h>
#include <analyze/count_contig_access_loops.h>
#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <analyze/find_indexing_loops.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen.h>
#include <container_utils.h>
#include <driver.h>
#include <lower.h>
#include <omp_utils.h>
#include <pass/const_fold.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
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
#include <schedule/permute.h>
#include <schedule/pluto_fuse.h>
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

Stmt Schedule::quickOptimizations(const Stmt &_ast) {
    auto ast = _ast;
    ast = constFold(ast);
    ast = makeReduction(ast);
    ast = flattenStmtSeq(ast);
    return ast;
}

Schedule::Schedule(const Stmt &ast, int verbose)
    : verbose_(verbose), memoized_(Ref<MemoizedSchedules>::make()),
      rng_(Ref<OpenMPRandomEngine>::make(0)) /* TODO: set seed */,
      randCtx_(Ref<RandCtx<OpenMPRandomEngine>>::make(*rng_)) {
    openTrans_.emplace_back(quickOptimizations(ast), ScheduleLog());
}

void Schedule::beginTransaction() { openTrans_.emplace_back(ast(), logs()); }

void Schedule::commitTransaction() {
    if (openTrans_.size() == 1) {
        ERROR(
            "The outer-most default transaction does not need to be committed");
    }
    auto trans = std::move(openTrans_.back());
    openTrans_.pop_back();
    if (verbose_ >= 2) {
        auto &&os = logger();
        os << "Committing schedule(s): ";
        auto logs = asVector(openTrans_.back().logs_, trans.logs_);
        for (auto &&[i, item] : views::enumerate(logs)) {
            os << (i > 0 ? ", " : "") << *item;
        }
        os << ", resulting in:" << std::endl << trans.ast_ << std::endl;
    }
    openTrans_.back() = std::move(trans);
}

void Schedule::abortTransaction() {
    if (openTrans_.size() == 1) {
        ERROR("The outer-most default transaction cannot be aborted");
    }
    openTrans_.pop_back();
}

const Stmt &Schedule::ast() const { return openTrans_.back().ast_; }
void Schedule::setAst(const Stmt &ast) { openTrans_.back().ast_ = ast; }

const ScheduleLog &Schedule::logs() const { return openTrans_.back().logs_; }
void Schedule::setLogs(const ScheduleLog &logs) {
    openTrans_.back().logs_ = logs;
}

// Make a log item with specifc parameter and result types
#define MAKE_LOG(TYPE, FUNC, ...)                                              \
    ([this](const auto &func, const auto &_params) {                           \
        auto params = getPackFromID(this, _params);                            \
        /* decay is required: we must not store an reference */                \
        typedef ScheduleLogItemImpl<                                           \
            ScheduleType::TYPE, std::decay_t<decltype(func)>,                  \
            std::decay_t<decltype(params)>,                                    \
            std::decay_t<decltype(std::apply(func, _params))>>                 \
            BaseClass;                                                         \
        class ScheduleLogItem##TYPE : public BaseClass {                       \
          public:                                                              \
            ScheduleLogItem##TYPE(const typename BaseClass::Invocable &f,      \
                                  const typename BaseClass::Params &p)         \
                : BaseClass(f, p) {}                                           \
        };                                                                     \
        return Ref<ScheduleLogItem##TYPE>::make(func, params);                 \
    })(futureSchedule(FUNC), std::make_tuple(__VA_ARGS__))

std::pair<ID, ID> Schedule::split(const ID &id, int factor, int nparts,
                                  int shift) {
    beginTransaction();
    auto log = appendLog(
        MAKE_LOG(Split, freetensor::split, id, factor, nparts, shift));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::reorder(const std::vector<ID> &order) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Reorder, freetensor::reorder, order));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

ID Schedule::merge(const ID &loop1, const ID &loop2) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Merge, freetensor::merge, loop1, loop2));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

std::vector<ID> Schedule::permute(
    const std::vector<ID> &loopsId,
    const std::function<std::vector<Expr>(std::vector<Expr>)> &transformFunc) {
    beginTransaction();
    //! FIXME: put this into schedule logs
    try {
        auto ret = freetensor::permute(ast(), loopsId, transformFunc);
        setAst(quickOptimizations(ret.first));
        commitTransaction();
        return ret.second;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw;
    }
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const ID &loop, FissionSide side, const ID &splitter,
                  const std::string &suffix0, const std::string &suffix1) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Fission, freetensor::fission, loop, side,
                                  splitter, suffix0, suffix1));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

ID Schedule::fuse(const ID &loop0, const ID &loop1, bool strict) {
    beginTransaction();
    auto log =
        appendLog(MAKE_LOG(Fuse, freetensor::fuse, loop0, loop1, strict));
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

void Schedule::swap(const std::vector<ID> &order) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Swap, freetensor::swap, order));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::blend(const ID &loop) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Blend, freetensor::blend, loop));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cache(const ID &stmt, const std::string &var, MemType mtype) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Cache, freetensor::cache, stmt, var, mtype));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cacheReduction(const ID &stmt, const std::string &var,
                         MemType mtype) {
    beginTransaction();
    auto log = appendLog(
        MAKE_LOG(CacheReduction, freetensor::cacheReduction, stmt, var, mtype));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::setMemType(const ID &def, MemType mtype) {
    beginTransaction();
    auto log =
        appendLog(MAKE_LOG(SetMemType, freetensor::setMemType, def, mtype));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::varSplit(const ID &def, int dim, VarSplitMode mode, int factor,
                        int nparts) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(VarSplit, freetensor::varSplit, def, dim,
                                  mode, factor, nparts));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::varMerge(const ID &def, int dim) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(VarMerge, freetensor::varMerge, def, dim));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::varReorder(const ID &def, const std::vector<int> &order) {
    beginTransaction();
    auto log =
        appendLog(MAKE_LOG(VarReorder, freetensor::varReorder, def, order));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

std::pair<ID, ID> Schedule::moveTo(const ID &_stmt, MoveToSide side,
                                   const ID &_dst) {
    beginTransaction();
    try {
        auto stmt = _stmt, dst = _dst;
        auto stmtBody = stmt;
        while (true) {
            Stmt s = findStmt(ast(), stmt);
            Stmt d = findStmt(ast(), dst);

            auto movingUp = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::Before;
                }
                if (auto prev = s->prevInCtrlFlow(); prev.isValid()) {
                    return d->isBefore(side == MoveToSide::After ? prev : s);
                } else {
                    return d->isBefore(s);
                }
            };
            auto movingDown = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::After;
                }
                if (auto next = s->nextInCtrlFlow(); next.isValid()) {
                    return (side == MoveToSide::Before ? next : s)->isBefore(d);
                } else {
                    return s->isBefore(d);
                }
            };

            if (movingUp()) {
                if (s->prevInCtrlFlow().isValid()) {
                    std::vector<ID> orderRev;
                    while (s->prevInCtrlFlow().isValid() && movingUp()) {
                        s = s->prevInCtrlFlow();
                        orderRev.emplace_back(s->id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<ID> order(orderRev.rbegin(), orderRev.rend());
                    swap(order);
                } else {
                    while (!s->prevInCtrlFlow().isValid() && movingUp()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast(), "Fission a " + toString(s->nodeType()) +
                                       " node in a StmtSeq is not currently "
                                       "supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    auto idMapBefore =
                        fission(s->id(), FissionSide::After, stmt, ".a", "")
                            .first;
                    stmtBody = idMapBefore.at(stmt);
                    stmt = idMapBefore.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else if (movingDown()) {
                if (s->nextInCtrlFlow().isValid()) {
                    std::vector<ID> order;
                    while (s->nextInCtrlFlow().isValid() && movingDown()) {
                        s = s->nextInCtrlFlow();
                        order.emplace_back(s->id());
                    }
                    order.emplace_back(stmt);
                    swap(order);
                } else {
                    while (!s->nextInCtrlFlow().isValid() && movingDown()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast(), "Fission a " + toString(s->nodeType()) +
                                       " node in a StmtSeq is not currently "
                                       "supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMapAfter =
                        fission(s->id(), FissionSide::Before, stmt, "", ".b")
                            .second;
                    stmtBody = idMapAfter.at(stmt);
                    stmt = idMapAfter.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else {
                commitTransaction();
                return {stmtBody, stmt};
            }
        }
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(ast(), "Invalid move_to(" + toString(_stmt) +
                                         ", " + toString(_dst) +
                                         "): " + e.what());
    }
}

void Schedule::inlining(const ID &def) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Inline, freetensor::inlining, def));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::parallelize(const ID &loop, const ParallelScope &parallel) {
    beginTransaction();
    auto log = appendLog(
        MAKE_LOG(Parallelize, freetensor::parallelize, loop, parallel));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::unroll(const ID &loop, bool immediate) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Unroll, freetensor::unroll, loop, immediate));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::vectorize(const ID &loop) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(Vectorize, freetensor::vectorize, loop));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::separateTail(bool noDuplicateVarDefs) {
    beginTransaction();
    auto log = appendLog(
        MAKE_LOG(SeparateTail, freetensor::separateTail, noDuplicateVarDefs));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::asMatMul(const ID &loop) {
    beginTransaction();
    auto log = appendLog(MAKE_LOG(AsMatMul, freetensor::asMatMul, loop));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

ID Schedule::plutoFuse(const ID &loop0, const ID &loop1, int nestLevel) {
    beginTransaction();
    auto log = appendLog(
        MAKE_LOG(PlutoFuse, freetensor::plutoFuse, loop0, loop1, nestLevel));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::autoSchedule(const Target &target, const Ref<RandTrace> &trace) {
    autoUseLib(target);
    autoFissionFuse(target, trace);
    autoReorder(target);
    autoParallelize(target);
    autoSetMemType(target);
    autoUnroll(target);
}

void Schedule::autoUseLib(const Target &target) {
    // Try to implement each top-level loops with lib calls
    for (auto &&_loop : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto loop = _loop.as<ForNode>();
        try {
            asMatMul(loop->id());
        } catch (const InvalidSchedule &e) {
            // If the loop is marked as preferLibs, we inline all local
            // variables, fission all the statments apart, and try applying to
            // each of them
            bool isPreferLibs = false;
            for (For l = loop;;) {
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
                     allDefs(loop, {AccessType::Cache})) {
                    try {
                        inlining(defId);
                    } catch (const InvalidSchedule &e) {
                        // do nothing
                    }
                }
                auto stmts =
                    allStmts(loop, {ASTNodeType::Store, ASTNodeType::ReduceTo});
                for (auto &&[i, stmt] : views::enumerate(stmts)) {
                    beginTransaction();
                    try {
                        fission(loop->id(), FissionSide::Before, stmt->id(),
                                "." + toString(i), "");
                        auto libStmtId =
                            fission(loop->id(), FissionSide::After, stmt->id(),
                                    "." + toString(i) + ".lib", "")
                                .first.at(loop->id());
                        asMatMul(libStmtId);
                        commitTransaction();
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                    }
                }
            }
        }
    }
}

void Schedule::autoReorder(const Target &target) {
    auto allLoops = findAllLoops(ast());
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop, DepDirection::Normal}});
    }

    // 0 = No dep
    // 1 = Reduction
    // 2 = Others
    std::unordered_map<ID, int> depLevel;
    FindDeps().direction(direction).ignoreReductionWAW(false)(
        ast(), [&](const Dependency &d) {
            ASSERT(d.dir_.size() == 1);
            auto &level = depLevel[d.dir_[0].first.id_];
            if (d.earlier()->nodeType() == ASTNodeType::ReduceTo &&
                d.later()->nodeType() == ASTNodeType::ReduceTo) {
                level = std::max(level, 1);
            } else {
                level = std::max(level, 2);
            }
        });

    std::function<void(For nest)> visitNest = [&, this](For nest) {
        // Currently we only reorder loops in a perfect loop nest
        std::vector<ID> perfectNest = {nest->id()};
        while (true) {
            if (auto inners =
                    findAll("<For><-(!<For><-)*#" + toString(nest->id()));
                inners.size() == 1) {
                nest = inners.front().as<ForNode>();
                perfectNest.emplace_back(nest->id());
            } else {
                break;
            }
        }

        auto sorted = perfectNest;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [&](const ID &lhs, const ID &rhs) {
                             return depLevel[lhs] < depLevel[rhs];
                         });
        if (sorted != perfectNest) {
            reorder(sorted);
        }

        for (auto &&subNest :
             findAll("<For><-(!<For><-)*#" + toString(nest->id()))) {
            visitNest(subNest.as<ForNode>());
        }
    };
    for (auto &&subNest : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        visitNest(subNest.as<ForNode>());
    }
}

void Schedule::autoFissionFuse(const Target &target,
                               const Ref<RandTrace> &trace) {
    RandCondStack conds;

    // Random decision on whether to fission or fuse:
    //
    // - Decision = 0: not to fuse, to fission
    // - Decision = 1: to fuse, not to fission
    //
    // Fusing (or not fissioning) may reduce parallelizing opportunities, which
    // is related to dependences on the two loops being fused (or the two
    // fissioned loops):
    //
    // - If neither loop has dependence: It doesn't matter
    // - If both loops have dependence: It doesn't matter, too
    // - If exactly one of the loop: It may have negative influence on
    // parallelizing
    //
    // Therefore, we add "the two loops having different
    // dependences" as a condition of our decision
    auto decisionId = PROGRAM_POSITION;
    auto decisionName = "fuse";
    auto depDiffCondName = "defDiff";

    // Record which loop is fission from which loop, so we don't fuse them back
    // to one
    //
    // We only care about the last fission. E.g., if we fission `L1 L2` to `L1
    // {L3 L4}` and then to `L1 {{L5 L6}, L4}`, we won't fuse L5 and L6 back,
    // but we can fuse L6 and L4
    std::unordered_map<ID, ID> fissionFrom;

    // Try to fission a loop into consecutive loops. Only fission at
    // beginnings of each sub-loops, and at each leaf nodes (Store, ReduceTo,
    // Eval)
    std::function<void(For nest)> tryFission = [&, this](For nest) {
        // Recurse first
        for (auto &&subNest :
             findAll("<For><-(!<For><-)*#" + toString(nest->id()))) {
            tryFission(subNest.as<ForNode>());
        }

        // Try fission
        auto thisId = nest->id();
        int partCnt = 0;
        auto splitters =
            findAll("(<For>|<Store>|<ReduceTo>|<Eval>)<-(!<For><-)*#" +
                    toString(nest->id()));
        for (auto &&[i, splitter] : views::enumerate(splitters)) {
            if (i == 0) {
                continue;
            }
            bool frontHasDep =
                FindDeps()
                    .direction({{{thisId, DepDirection::Different}}})
                    .filterSubAST(thisId)
                    .filterAccess([&](const AccessPoint &ap) {
                        return ap.stmt_->isBefore(splitter);
                    })
                    .exists(ast());
            bool backHasDep =
                FindDeps()
                    .direction({{{thisId, DepDirection::Different}}})
                    .filterSubAST(thisId)
                    .filterAccess([&](const AccessPoint &ap) {
                        return !ap.stmt_->isBefore(splitter);
                    })
                    .exists(ast());
            bool depDiff = frontHasDep != backHasDep;
            {
                RandCondGuard _(conds, depDiffCondName, depDiff);
                if (!randCtx_->decide(decisionId, decisionName, conds,
                                      {depDiff ? 1 : 0, depDiff ? 0 : 1}, trace,
                                      "not fission " + toString(thisId) +
                                          " before " +
                                          toString(splitter->id()) + "?")) {
                    beginTransaction();
                    try {
                        auto newId =
                            fission(thisId, FissionSide::Before, splitter->id(),
                                    "." + toString(partCnt), "")
                                .first.at(thisId);
                        fissionFrom[newId] = thisId;
                        partCnt++;
                        commitTransaction();
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                    }
                }
            }
        }
        fissionFrom[thisId] = thisId;
    };
    for (auto &&loop : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        tryFission(loop.as<ForNode>());
    }

    // Try to fuse each pair of consecutive loops, unless they are just
    // fissioned from the same loop
    std::function<void(Stmt)> tryFuse = [&, this](Stmt root) {
        For last;
        ID lastId;
        for (auto &&_loop :
             findAll("<For><-(!<For><-)*#" + toString(root->id()))) {
            auto loop = _loop.as<ForNode>();
            auto loopId = loop->id();
            if (!last.isValid()) {
                goto skip;
            }
            if (fissionFrom.count(loopId) && fissionFrom.count(lastId) &&
                fissionFrom.at(loopId) == fissionFrom.at(lastId)) {
                goto skip;
            }
            {
                bool thisHasDep =
                    FindDeps()
                        .direction({{{loopId, DepDirection::Different}}})
                        .filterSubAST(loopId)
                        .exists(ast());
                bool lastHasDep =
                    FindDeps()
                        .direction({{{lastId, DepDirection::Different}}})
                        .filterSubAST(lastId)
                        .exists(ast());
                bool depDiff = thisHasDep != lastHasDep;
                {
                    RandCondGuard _(conds, depDiffCondName, depDiff);
                    if (randCtx_->decide(decisionId, decisionName, conds,
                                         {depDiff ? 1 : 0, depDiff ? 0 : 1},
                                         trace,
                                         "fuse " + toString(lastId) + " and " +
                                             toString(loopId) + "?")) {
                        beginTransaction();
                        try {
                            try {
                                lastId =
                                    moveTo(lastId, MoveToSide::Before, loopId)
                                        .first;
                            } catch (const InvalidSchedule &e) {
                                loopId =
                                    moveTo(loopId, MoveToSide::After, lastId)
                                        .first;
                            }
                            loopId = fuse(lastId, loopId, true);
                            loop = find(loopId).as<ForNode>();
                            commitTransaction();
                        } catch (const InvalidSchedule &e) {
                            abortTransaction();
                            tryFuse(last);
                        }
                    }
                }
            }
        skip:
            lastId = loopId, last = loop;
        }
        if (last.isValid()) {
            tryFuse(last);
        }
    };
    tryFuse(ast());
}

void Schedule::autoParallelize(const Target &target) {
#ifdef FT_WITH_CUDA
    // [GPU only] Try to parallelize loops accessing contiguous items as warps
    if (target.type() == TargetType::GPU) {
        // We try to parallelize the loop with most contiguous access count
        // first. If the counts are equal, we try to parallel the out-most loop
        // with the same count first
        CountContigAccessLoops contigFinder;
        contigFinder(ast());
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

            beginTransaction();
            try {
                auto [l0, l1] =
                    split(loop->id(), ((GPUTarget &)target).warpSize());
                parallelize(l1, threadIdxX);

                try {
                    // Reorder this scope to as outer as possible
                    auto refCntHolder = ast();
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
                commitTransaction();
            } catch (const InvalidSchedule &e) {
                abortTransaction();
            }
        }
    }
#endif // FT_WITH_CUDA

    // Try to merge and parallelize as many outer loops as possible
    std::function<void(For)> autoParallelizeOuter = [&](For root) {
#ifdef FT_WITH_CUDA
        bool parentIsWarp = false;
        while (root->property_->parallel_ != serialScope) {
            if (auto inners =
                    findAll("<For><-(!<For><-)*#" + toString(root->id()));
                inners.size() == 1) {
                root = inners.front().as<ForNode>();
                parentIsWarp = true;
            } else {
                break;
            }
        }
#endif // FT_WITH_CUDA

        // Count how many loops we can merge and drop the result. Don't
        // worry about repeatly doing the same merging, because we have
        // memoized schedules
        int maxMergeLevel = 0;
        beginTransaction();
        try {
            ID mergedId;
            auto loop = root;
            while (true) {
                ID loopId = loop->id();
                if (find(loopId).as<ForNode>()->property_->parallel_ !=
                    serialScope) {
                    break;
                }
                mergedId =
                    mergedId.isValid() ? merge(mergedId, loopId) : loopId;
                maxMergeLevel++;
                if (auto inners =
                        findAll("<For><-(!<For><-)*#" + toString(loopId));
                    inners.size() == 1) {
                    loop = inners.front().as<ForNode>();
                } else {
                    break;
                }
            }
        } catch (const InvalidSchedule &e) {
            // do nothing
        }
        abortTransaction();

        // Suppose we can merge n loops at maximum, we try merging and
        // parallelizing n loops first, then try n - 1, n - 2, and so on.
        bool done = false;
        for (int mergeLevel = maxMergeLevel; mergeLevel > 0; mergeLevel--) {
            beginTransaction();
            try {
                ID mergedId;
                auto loop = root;
                for (int i = 0; i < mergeLevel; i++) {
                    ID loopId = loop->id();
                    mergedId =
                        mergedId.isValid() ? merge(mergedId, loopId) : loopId;
                    if (i + 1 < mergeLevel) {
                        loop = find("<For><-(!<For><-)*#" + toString(loopId))
                                   .as<ForNode>();
                    }
                }

                switch (target.type()) {
                case TargetType::CPU:
                    parallelize(mergedId, OpenMPScope{});
                    break;

#ifdef FT_WITH_CUDA
                case TargetType::GPU: {
                    auto merged = find(mergedId);
                    auto isParallelLoop = [](const Stmt &s) {
                        return s->nodeType() == ASTNodeType::For &&
                               s.as<ForNode>()->property_->parallel_ !=
                                   serialScope;
                    };
                    bool childIsWarp =
                        !findAllStmt(merged, isParallelLoop).empty();
                    // We guarantee the following requirements in order:
                    // 1. make sure all SMs are used
                    // 2. if there are enough threads, make sure blockDim is
                    // not too large If the loop length is constant, we
                    // split it only once, to reduce redundant guards, and
                    // save time for dependency analysis. If not, we split
                    // it twice, and merge once
                    int numSM = ((GPUTarget &)target).multiProcessorCount();
                    int maxThreads = 256; // Can be max thread per block (1024),
                                          // but our generated kernels are huge,
                                          // so set it lower to reserve for more
                                          // registers. TODO: no magic number
                    if (parentIsWarp || childIsWarp) {
                        maxThreads /= ((GPUTarget &)target).warpSize();
                    }
                    ID l1, l1b, l2;
                    if (auto loopNode = merged.as<ForNode>();
                        loopNode->len_->nodeType() == ASTNodeType::IntConst) {
                        auto len = loopNode->len_.as<IntConstNode>()->val_;
                        if (len < numSM * maxThreads) {
                            std::tie(l1, l2) = split(mergedId, -1, numSM);
                        } else {
                            std::tie(l1, l2) = split(mergedId, maxThreads);
                        }
                    } else {
                        // We don't use the `nparts` mode of `split`,
                        // because it will hinder dependency analysis.
                        // Instead, we use the `factor` mode and then
                        // reorder. See the doc string of `split` for
                        // details
                        std::tie(l2, l1) = split(mergedId, numSM);
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
                            parallelize(l1b, blockIdxX);
                        } else {
                            parallelize(l1, blockIdxX);
                        }
                    }
                    if (!findAll(l2).empty()) {
                        parallelize(l2, (!parentIsWarp && !childIsWarp)
                                            ? threadIdxX
                                            : threadIdxY);
                    }
                    break;
                }

#endif // FT_WITH_CUDA
                default:
                    ASSERT(false);
                }

                done = true;
                commitTransaction();
                break;
            } catch (const InvalidSchedule &e) {
                abortTransaction();
            }
        }

        if (!done) {
            for (auto &&subLoop :
                 findAll("<For><-(!<For><-)*#" + toString(root->id()))) {
                autoParallelizeOuter(subLoop.as<ForNode>());
            }
        }
    };
    for (auto &&_root : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto root = _root.as<ForNode>();
        // If the outer most loop is too short, we try the second outer loops
        // instead
        if (auto &&inners =
                findAll("<For><-(!<For><-)*#" + toString(root->id()));
            inners.size() > 1 &&
            root->len_->nodeType() == ASTNodeType::IntConst &&
            root->len_.as<IntConstNode>()->val_ < 32) {
            for (auto &&inner : inners) {
                autoParallelizeOuter(inner.as<ForNode>());
            }
        } else {
            autoParallelizeOuter(root);
        }
    }
}

void Schedule::autoSetMemType(const Target &target) {
    // Try to put each VarDef as near to processor as possible
    if (target.type() == TargetType::GPU) {
        for (auto &&[defId, name] : allDefs(ast(), {AccessType::Cache})) {
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
        for (auto &&[loop, defs] : findIndexingLoops(ast())) {
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
    for (auto &&_loop : findAll("<For>")) {
        auto loop = _loop.as<ForNode>();
        if (loop->property_->parallel_ == serialScope &&
            !loop->property_->vectorize_ && !loop->property_->unroll_ &&
            loop->len_->nodeType() == ASTNodeType::IntConst &&
            loop->len_.as<IntConstNode>()->val_ <= 4) {
            unroll(loop->id());
        }
    }
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
            std::vector<Ref<Driver>> drivers(batchSize);
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
                    drivers[j] = Ref<Driver>::make(lowered, code, device);
                },
                omp_sched_static); // use schedule(static) to guarantee
                                   // deterministic RNG
            for (int j = 0; j < batchSize; j++) {
                auto &d = *drivers[j];
                auto &[trace, _1, _2, t, stddev] = trials[i * batchSize + j];
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
        throw;
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
