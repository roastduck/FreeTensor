#include <analyze/count_contig_access_loops.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <math/utils.h>
#include <pass/const_fold.h>
#include <pass/simplify.h>
#include <schedule.h>

namespace freetensor {

template <typename T>
std::optional<T> optMul(const std::optional<T> &lhs,
                        const std::optional<T> &rhs) {
    if (lhs.has_value() && rhs.has_value()) {
        return *lhs * *rhs;
    } else {
        return std::nullopt;
    }
}

static std::pair<std::unordered_set<ID>, std::unordered_set<ID>>
allParallelizableLoops(const Schedule &s) {
    std::unordered_set<ID> allCandidates;
    std::vector<FindDepsDir> direction;
    for (auto &&loop : s.findAll("<For>")) {
        if (loop.as<ForNode>()->property_->parallel_ != serialScope) {
            continue; // Already parallelized
        }
        allCandidates.emplace(loop->id());
        FindDepsDir dirItem{{loop->id(), DepDirection::Normal}};
        for (auto &&outerLoop : s.findAll("<For>->>" + toString(loop->id()))) {
            dirItem.push_back({outerLoop->id(), DepDirection::Same});
        }
        direction.emplace_back(std::move(dirItem));
    }
    FindDeps().direction(direction)(s.ast(), [&](const Dependence &d) {
        allCandidates.erase(d.dir_[0].first.id_);
    });
    auto noRedCandidates = allCandidates;
    FindDeps().direction(direction).ignoreReductionWAW(false)(
        s.ast(), [&](const Dependence &d) {
            noRedCandidates.erase(d.dir_[0].first.id_);
        });
    return {allCandidates, noRedCandidates};
}

static std::optional<int64_t> findStaticLen(const Schedule &s,
                                            const ID &loopId) {
    if (auto &&len = constFold(s.find(loopId).as<ForNode>()->len_);
        len->nodeType() == ASTNodeType::IntConst) {
        return len.as<IntConstNode>()->val_;
    } else {
        return std::nullopt;
    }
}

static ID mergeAll(Schedule &s, const auto &loops) {
    ID mergedId;
    for (auto &&loopId : loops) {
        if (!mergedId.isValid()) {
            mergedId = loopId;
        } else {
            if (auto &&middle =
                    s.findAll("(<For><<-" + toString(mergedId) + ")&(<For>->>" +
                              toString(loopId) + ")");
                !middle.empty()) {
                // Maybe some loops from warp parallelizing, in outer to
                // inner order. Reorder them all inner
                std::vector<ID> order = {loopId};
                for (auto &&item : middle) {
                    order.emplace_back(item->id());
                }
                s.reorder(order);
            }
            mergedId = s.merge(mergedId, loopId);
        }
    }
    return mergedId;
}

static bool testParallelizableAfterMerge(Schedule &s, const auto &loops,
                                         const ParallelScope &scope) {
    s.beginTransaction();
    try {
        s.parallelize(mergeAll(s, loops), scope);
    } catch (const InvalidSchedule &e) {
        s.abortTransaction();
        return false;
    }
    s.abortTransaction();
    return true;
}

/**
 * Try to parallelize a perfect loop nest consisting of loops in `loops`
 *
 * Loops are parallelized to different scopes with respect to different
 * priorities
 *
 * Static-lengthed and dynammic-length loops are treated differently:
 *
 * - Static-lengthed loops are splitted for each parallel scope separatedly, so
 * the tile will be aligned with the original loops. The following figure shows
 * an example, where the dimension rows and columns refer to two loops, and each
 * number refer to one execution unit (or thread)
 *
 * 0000111
 * 2222333
 *
 * - Dynamic-lengthed loops are first merged together before splitted for each
 * parallel scope. This is a last resort, which may lead to unaligned tiles,
 * like in the following figure. This unalignment may lead to poor runtime
 * locality and/or long compiling time for analyzing complex index expressions
 *
 * 0000111
 * 1222233
 *
 * We use a greedy algorithm that is guaranteed to work when the number of
 * scopes is no more than 3, but may fail for more scopes.
 *
 * Algorithm:
 *
 * 1. Find a scope that all scopes more outer than it are of equal or more
 * priority, and all scopes more inner than it are of less priority, or vice
 * versa (always possible when # of scopes <= 3).
 * 2. Do a `factor` split if the inner loops are prioritized, or a `nparts`
 * split if the outer ones are prioritized. If all the affected loops have
 * static length, split only one of them to avoid misalignment. Otherwise, merge
 * them all first and then split.
 * 3. Recurse into each side respectively (goto 1).
 *
 * @param loops : Loops from outer to inner
 * @param scopes : Scopes we want to parallelize the loops to. E.g.,
 * ["blockIdx.x", "threadIdx.y", "threadIdx.x"]. Outer loops prefer left-side
 * scopes in this list. A scope can be `serialScope` for not parallelizing
 * @param limits : The maximum length acceptable by each scope in `scopes`.
 * `nullopt` = no limit. One of the limits should be no limit to absorb exceeded
 * length
 * @param priority : Priority of each scope. E.g., if `scopes = ["blockIdx.x",
 * "threadIdx.y", "threadIdx.x"]`, `priority = [10, 20, 0]`, we fill
 * `threadIdx.y` to reach its limits first, and then `blockIdx.x`, and then
 * `threadIdx.x`.
 */
static void parallelizePerfectNest(
    Schedule &s, const std::vector<ID> &loops,
    const std::vector<ParallelScope> &scopes,
    const std::vector<std::optional<int64_t>> &limits,
    const std::vector<int> &priority,
    const std::function<void(const ParallelScope &,
                             const std::optional<int64_t> &)> &callback) {
    ASSERT(!scopes.empty());
    if (scopes.size() == 1) {
        if (scopes[0] != serialScope) {
            s.beginTransaction();
            try {
                auto &&mergedId = mergeAll(s, loops);
                if (mergedId.isValid()) {
                    s.parallelize(mergedId, scopes[0]);
                    if (callback != nullptr) {
                        callback(scopes[0], findStaticLen(s, mergedId));
                    }
                }
                s.commitTransaction();
            } catch (const InvalidSchedule &e) {
                s.abortTransaction();
            }
        }
        return;
    }

    std::vector<std::optional<int64_t>> lengths;
    lengths.reserve(loops.size());
    for (auto &&loopId : loops) {
        lengths.emplace_back(findStaticLen(s, loopId));
    }

    // Check if we can found a split point where inner parts are prioritized.
    // Check this before checking outer parts because `factor` is preferred than
    // `nparts` (see doc for `split`)
    for (int i = 1, n = scopes.size(); i < n; i++) {
        if (*std::min_element(priority.begin() + i, priority.end()) >
            *std::max_element(priority.begin(), priority.begin() + i)) {
            auto scopesL = ranges::to<std::vector>(scopes | views::take(i));
            auto limitsL = ranges::to<std::vector>(limits | views::take(i));
            auto priorityL = ranges::to<std::vector>(priority | views::take(i));
            auto scopesR = ranges::to<std::vector>(scopes | views::drop(i));
            auto limitsR = ranges::to<std::vector>(limits | views::drop(i));
            auto priorityR = ranges::to<std::vector>(priority | views::drop(i));

            auto lim =
                std::accumulate(limits.begin() + i, limits.end(),
                                std::optional<int64_t>{1}, optMul<int64_t>);
            if (!lim.has_value()) {
                parallelizePerfectNest(s, loops, scopesR, limitsR, priorityR,
                                       callback);
                return;
            }
            ASSERT(*lim > 0);
            int m = loops.size(), j = m - 1;
            while (j >= 0 && lengths[j].has_value() && *lim >= *lengths[j]) {
                ASSERT(*lengths[j] > 0);
                *lim = ceilDiv(*lim, *lengths[j--]);
            }
            if (j == -1 || lim == 1) {
                parallelizePerfectNest(
                    s, ranges::to<std::vector>(loops | views::take(j + 1)),
                    scopesL, limitsL, priorityL, callback);
                parallelizePerfectNest(
                    s, ranges::to<std::vector>(loops | views::drop(j + 1)),
                    scopesR, limitsR, priorityR, callback);
            } else if (lengths[j].has_value()) {
                auto &&[outerId, innerId] = s.split(loops[j], *lim);
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        loops | views::take(j), views::single(outerId))),
                    scopesL, limitsL, priorityL, callback);
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        views::single(innerId), loops | views::drop(j + 1))),
                    scopesR, limitsR, priorityR, callback);
            } else {
                auto &&[outerId, innerId] =
                    s.split(mergeAll(s, loops | views::take(j + 1)), *lim);
                if (!s.findAll(outerId).empty()) { // Maybe null by assertions
                    parallelizePerfectNest(s, {outerId}, scopesL, limitsL,
                                           priorityL, callback);
                }
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        views::single(innerId), loops | views::drop(j + 1))),
                    scopesR, limitsR, priorityR, callback);
            }
            return;
        }
    }

    // Check if we can found a split point where outer parts are prioritized.
    for (int i = 1, n = scopes.size(); i < n; i++) {
        if (*std::min_element(priority.begin(), priority.begin() + i) >
            *std::max_element(priority.begin() + i, priority.end())) {
            auto scopesL = ranges::to<std::vector>(scopes | views::take(i));
            auto limitsL = ranges::to<std::vector>(limits | views::take(i));
            auto priorityL = ranges::to<std::vector>(priority | views::take(i));
            auto scopesR = ranges::to<std::vector>(scopes | views::drop(i));
            auto limitsR = ranges::to<std::vector>(limits | views::drop(i));
            auto priorityR = ranges::to<std::vector>(priority | views::drop(i));

            auto lim =
                std::accumulate(limits.begin(), limits.begin() + i,
                                std::optional<int64_t>{1}, optMul<int64_t>);
            if (!lim.has_value()) {
                parallelizePerfectNest(s, loops, scopesL, limitsL, priorityL,
                                       callback);
                return;
            }
            ASSERT(*lim > 0);
            int m = loops.size(), j = 0;
            while (j < m && lengths[j].has_value() && *lim >= *lengths[j]) {
                ASSERT(*lengths[j] > 0);
                *lim = ceilDiv(*lim, *lengths[j++]);
            }
            if (j == m || lim == 1) {
                parallelizePerfectNest(
                    s, ranges::to<std::vector>(loops | views::take(j)), scopesL,
                    limitsL, priorityL, callback);
                parallelizePerfectNest(
                    s, ranges::to<std::vector>(loops | views::drop(j)), scopesR,
                    limitsR, priorityR, callback);
            } else if (lengths[j].has_value()) {
                auto &&[outerId, innerId] = s.split(loops[j], -1, *lim);
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        loops | views::take(j), views::single(outerId))),
                    scopesL, limitsL, priorityL, callback);
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        views::single(innerId), loops | views::drop(j + 1))),
                    scopesR, limitsR, priorityR, callback);
            } else {
                auto &&[outerId, innerId] =
                    s.split(mergeAll(s, loops | views::drop(j)), -1, *lim);
                parallelizePerfectNest(
                    s,
                    ranges::to<std::vector>(views::concat(
                        loops | views::take(j), views::single(outerId))),
                    scopesL, limitsL, priorityL, callback);
                if (!s.findAll(innerId).empty()) { // Maybe null by assertions
                    parallelizePerfectNest(s, {innerId}, scopesR, limitsR,
                                           priorityR, callback);
                }
            }
            return;
        }
    }

    ERROR("The priority is unsupported by the current greedy algorithm");
}

void Schedule::autoParallelize(const Ref<Target> &target) {
#ifdef FT_WITH_CUDA
    // [GPU only] Try to parallelize loops accessing contiguous items as warps
    if (target->type() == TargetType::GPU) {
        // We try to parallelize the loop with most contiguous access count
        // first. If the counts are equal, we try to parallel the out-most loop
        // with the same count first
        auto simplified =
            simplify(ast()); // Simplify lengths and remove 1-lengthed loops
        CountContigAccessLoops contigFinder;
        contigFinder(simplified);
        auto ignoreIfTooShort =
            [&](const std::pair<ID, std::pair<int64_t, int>> &item) {
                auto loop = findStmt(simplified, item.first);
                if (auto &&len = loop.as<ForNode>()->len_;
                    len->nodeType() == ASTNodeType::IntConst &&
                    len.as<IntConstNode>()->val_ <= 4) {
                    return false;
                }
                return true;
            };
        auto contigLoops =
            ranges::to<std::vector<std::pair<ID, std::pair<int64_t, int>>>>(
                contigFinder.counts() | views::filter(ignoreIfTooShort));
        std::sort(contigLoops.begin(), contigLoops.end(),
                  [](const std::pair<ID, std::pair<int64_t, int>> &lhs,
                     const std::pair<ID, std::pair<int64_t, int>> &rhs) {
                      return lhs.second > rhs.second;
                  });
        for (auto &&[loopId, cnt] : contigLoops) {
            beginTransaction();
            try {
                auto [l0, l1] =
                    split(loopId, target.as<GPUTarget>()->warpSize());
                parallelize(l1, threadIdxX);

                try {
                    // Reorder this scope outer if the outer loop carries
                    // reduction, to make it possible to do thread-local
                    // reduction (maybe by `cache_reduction` in the future),
                    // e.g.:
                    //
                    // s = 0
                    // for p.1  --> Original inner loop
                    //   s.local = 0
                    //   for p.0
                    //     s.local += ...
                    //   s += s.local
                    auto refCntHolder = ast();
                    auto c = find(l1);
                    if (c->parentStmt().isValid()) {
                        for (c = c->parentStmt(); c->parentStmt().isValid();
                             c = c->parentStmt()) {
                            if (c->nodeType() == ASTNodeType::For) {
                                if (!FindDeps()
                                         .direction({{{c->id(),
                                                       DepDirection::Normal}}})
                                         .ignoreReductionWAW(false)
                                         .filterSubAST(c->id())
                                         .exists(ast())) {
                                    break;
                                }
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

    auto &&[_allCandidates, _noRedCandidates] = allParallelizableLoops(*this);
    auto &&allCandidates = _allCandidates;
    auto &&noRedCandidates = _noRedCandidates;

    // Try to merge and parallelize as many outer loops as possible
    std::function<void(For, bool, bool)> autoParallelizeOuter =
        [&](For root, bool parentIsWarp, bool parallelizeAmongBlocks) {
            // We first try to parallel with best effort. But after we have
            // reached enough degree of parallelism, we stop introducing
            // parallel reductions, and only parallelize truly dependence-free
            // loops. Since we have already reorder dependence-free loops out of
            // reduction loops, we can safely stop on the first success without
            // lossing the possiblily of parallelizing any dependence-free loop.

            // I. Collect paralleliable loops
            std::vector<ID> localNoRed, localAll;
            ParallelScope testScope;
            switch (target->type()) {
            case TargetType::CPU:
                testScope = OpenMPScope{};
                break;
#ifdef FT_WITH_CUDA
            case TargetType::GPU:
                testScope = blockIdxX;
                break;
#endif // FT_WITH_CUDA
            }
            for (ID id = root->id();;) {
                if (allCandidates.count(id) &&
                    testParallelizableAfterMerge(
                        *this, views::concat(localAll, views::single(id)),
                        testScope)) {
                    localAll.emplace_back(id);
                }
                if (noRedCandidates.count(id) &&
                    testParallelizableAfterMerge(
                        *this, views::concat(localNoRed, views::single(id)),
                        testScope)) {
                    localNoRed.emplace_back(id);
                }
                if (auto nexts = findAll("<For><-(!<For><-)*" + toString(id));
                    nexts.size() == 1) {
                    id = nexts.front()->id();
                } else {
                    break;
                }
            }

            // II. Collect target degree of parallelism
            int numCPU = 0, numSM = 0, maxThreads = 0;
            switch (target->type()) {
            case TargetType::CPU:
                numCPU = target.as<CPUTarget>()->nCores();
                break;
#ifdef FT_WITH_CUDA
            case TargetType::GPU: {
                numSM = target.as<GPUTarget>()->multiProcessorCount();
                maxThreads =
                    256; // Can be max thread per block (1024), but our
                         // generated kernels are huge, so set it lower to
                         // reserve for more registers. TODO: no magic number
                if (parentIsWarp &&
                    findAll("<For><-(!<For><-)*" + toString(root->id()))
                            .size() > 1) {
                    // There are multiple loop nests inside one kernel. Don't
                    // parallelize these inner loops to blocks. Otherwise, there
                    // might be cross-block dependence inside one kernel, which
                    // cannot be resolved.
                    parallelizeAmongBlocks = false;
                }
                bool childIsWarp =
                    !findAllStmt(root, [](const Stmt &s) {
                         return s->nodeType() == ASTNodeType::For &&
                                s.as<ForNode>()->property_->parallel_ !=
                                    serialScope;
                     }).empty();
                if (parentIsWarp || childIsWarp) {
                    maxThreads /= target.as<GPUTarget>()->warpSize();
                }
                break;
            }
#endif // FT_WITH_CUDA
            }

            // III. Do parallelization
            bool done = false;
            std::vector<ParallelScope> scopes;
            std::vector<std::optional<int64_t>> limits;
            std::vector<int> priority;

            // III a. No reduction
            switch (target->type()) {
            case TargetType::CPU:
                scopes = {OpenMPScope{}};
                limits = {std::nullopt};
                priority = {0};
                break;

#ifdef FT_WITH_CUDA
            case TargetType::GPU:
                if (parallelizeAmongBlocks) {
                    scopes = {
                        blockIdxY, // Finally use more blocks
                        blockIdxX, // First occupy all SM
                        threadIdxY // Next fill each SM (threadIdxX is reserved
                                   // for warps)
                    };
                    limits = {
                        std::nullopt,
                        numSM,
                        maxThreads,
                    };
                    priority = {0, 2, 1};

                    // Have a try to see if we don't need blockIdxY for simpler
                    // code
                    bool smFull = false;
                    beginTransaction();
                    parallelizePerfectNest(
                        *this, localNoRed, scopes, limits, priority,
                        [&](const ParallelScope &scope,
                            const std::optional<int64_t> &len) {
                            if (scope == blockIdxY && len.has_value()) {
                                smFull = true;
                            }
                        });
                    abortTransaction();
                    if (smFull) {
                        scopes = {blockIdxX, threadIdxY};
                        limits = {std::nullopt, maxThreads};
                        priority = {0, 1};
                    }
                } else {
                    scopes = {threadIdxY};
                    limits = {std::nullopt};
                    priority = {0};
                }
                break;
#endif // FT_WITH_CUDA
            }
            bool needParRed = true;
            beginTransaction();
            parallelizePerfectNest(
                *this, localNoRed, scopes, limits, priority,
                [&](const ParallelScope &scope,
                    const std::optional<int64_t> &len) {
                    done = true;
                    if (std::holds_alternative<OpenMPScope>(scope) &&
                        len.has_value() && *len >= numCPU) {
                        needParRed = false;
                    }
#ifdef FT_WITH_CUDA
                    if ((scope == blockIdxY && len.has_value()) ||
                        (scope == blockIdxX && len.has_value() &&
                         *len >= numSM)) {
                        needParRed = false;
                    }
#endif // FT_WITH_CUDA
                });

            // III b. Reduction
            if (!needParRed) {
                commitTransaction();
            } else {
                abortTransaction();
                switch (target->type()) {
                case TargetType::CPU:
                    scopes = {OpenMPScope{}, serialScope};
                    limits = {numCPU, std::nullopt};
                    priority = {1, 0};
                    break;
#ifdef FT_WITH_CUDA
                case TargetType::GPU:
                    if (parallelizeAmongBlocks) {
                        scopes = {
                            blockIdxZ,   // Next try to use more blocks
                            threadIdxZ,  // First fill each SM because we pre
                                         // reducing inside an SM
                            serialScope, // Finally do serial reduction
                        };
                        limits = {numSM, maxThreads, std::nullopt};
                        priority = {1, 2, 0};
                    } else {
                        scopes = {threadIdxZ, serialScope};
                        limits = {maxThreads, std::nullopt};
                        priority = {1, 0};
                    }
                    break;
#endif // FT_WITH_CUDA
                }
                parallelizePerfectNest(*this, localAll, scopes, limits,
                                       priority,
                                       [&](auto &&, auto &&) { done = true; });
            }

            /// IV. Recurse into sub-loops if failed
            if (!done) {
                for (auto &&subLoop :
                     findAll("<For><-(!<For><-)*" + toString(root->id()))) {
                    autoParallelizeOuter(subLoop.as<ForNode>(),
                                         parentIsWarp ||
                                             root->property_->parallel_ !=
                                                 serialScope,
                                         parallelizeAmongBlocks);
                }
            }
        };
    for (auto &&_root : findAll("<For><-(!<For><-)*<-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto root = _root.as<ForNode>();
        // If the outer most loop is too short, we try the second outer loops
        // instead
        if (auto &&inners =
                findAll("<For><-(!<For><-)*" + toString(root->id()));
            inners.size() > 1 &&
            root->len_->nodeType() == ASTNodeType::IntConst &&
            root->len_.as<IntConstNode>()->val_ < 32) {
            for (auto &&inner : inners) {
                autoParallelizeOuter(inner.as<ForNode>(), false, true);
            }
        } else {
            autoParallelizeOuter(root, false, true);
        }
    }
}

} // namespace freetensor
