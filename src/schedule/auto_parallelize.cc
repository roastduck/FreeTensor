#include <analyze/count_contig_access_loops.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <pass/const_fold.h>
#include <pass/simplify.h>
#include <schedule.h>

namespace freetensor {

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
    std::function<void(For, bool, bool, int64_t)> autoParallelizeOuter =
        [&](For root, bool parentIsWarp, bool parallelizeAmongBlocks,
            int64_t outerParallelSize) {
#ifdef FT_WITH_CUDA
            if (root->property_->parallel_ != serialScope) {
                parentIsWarp = true;
                if (findAll("<For><-(!<For><-)*" + toString(root->id()))
                        .size() > 1) {
                    // There are multiple loop nests inside one kernel. Don't
                    // parallelize these inner loops to blocks. Otherwise, there
                    // might be cross-block dependence inside one kernel, which
                    // cannot be resolved.
                    parallelizeAmongBlocks = false;
                }
                // We will find out `maxMergeLevel == 0`, and recurse into the
                // next frame of `autoParallelizeOuter`.
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
                            findAll("<For><-(!<For><-)*" + toString(loopId));
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
            Expr mergedLen;
            for (int mergeLevel = maxMergeLevel; mergeLevel > 0; mergeLevel--) {
                beginTransaction();
                try {
                    ID mergedId;
                    auto loop = root;
                    for (int i = 0; i < mergeLevel; i++) {
                        ID loopId = loop->id();
                        mergedId = mergedId.isValid() ? merge(mergedId, loopId)
                                                      : loopId;
                        if (i + 1 < mergeLevel) {
                            loop = find("<For><-(!<For><-)*" + toString(loopId))
                                       .as<ForNode>();
                        }
                    }
                    Stmt merged = find(mergedId);
                    ASSERT(merged->nodeType() == ASTNodeType::For);
                    mergedLen = constFold(merged.as<ForNode>()->len_);

                    switch (target->type()) {
                    case TargetType::CPU:
                        parallelize(mergedId, OpenMPScope{});
                        break;

#ifdef FT_WITH_CUDA
                    case TargetType::GPU: {
                        auto isParallelLoop = [](const Stmt &s) {
                            return s->nodeType() == ASTNodeType::For &&
                                   s.as<ForNode>()->property_->parallel_ !=
                                       serialScope;
                        };
                        bool childIsWarp =
                            !findAllStmt(merged, isParallelLoop).empty();
                        // We are parallelizing loops to occupy SMs and cores in
                        // each SM. Depending on wether there is dependence
                        // along the loop (only reduction dependence is
                        // allowed), we have different priority.
                        //
                        // If there IS NO dependence, we want to occupy as much
                        // hardware resouce as possible. As the loop length
                        // increases, we meet the following requirements in
                        // order:
                        //
                        // 1. If `parallelizeAmongBlocks`, we try to use all SMs
                        // by increasing the number of blocks, else use only one
                        // SM (one block).
                        // 2. Try to use more threads on each SM. But if there
                        // are too many, scale to more blocks. There extra
                        // blocks will be queued to run on SMs.
                        //
                        // If there IS dependence, we want to put threads near
                        // each other if there are few threads. We only need to
                        // meet the following requirement:
                        //
                        // 1. Try to allocate threads on one SM (one block).
                        // Only if the number of threads reaches the limits of
                        // one block and `parallelizeAmongBlocks` is true, scale
                        // to more blocks.
                        //
                        // When splitting a loop, if the loop length is
                        // constant, we split it only once, to reduce redundant
                        // guards, and save time for dependence analysis. If
                        // not, we split it twice, and merge once
                        int numSM = 1;
                        if (parallelizeAmongBlocks) {
                            numSM =
                                target.as<GPUTarget>()->multiProcessorCount();
                        }
                        int maxThreads =
                            256; // Can be max thread per block (1024),
                                 // but our generated kernels are huge,
                                 // so set it lower to reserve for more
                                 // registers. TODO: no magic number
                        if (parentIsWarp || childIsWarp) {
                            maxThreads /= target.as<GPUTarget>()->warpSize();
                        }
                        bool hasDep =
                            FindDeps()
                                .direction(
                                    {{{mergedId, DepDirection::Different}}})
                                .filterSubAST(mergedId)
                                .ignoreReductionWAW(false)
                                .exists(ast());
                        ID l1, l1b, l2;
                        if (!hasDep) {
                            if (auto loopNode = merged.as<ForNode>();
                                loopNode->len_->nodeType() ==
                                ASTNodeType::IntConst) {
                                auto len =
                                    loopNode->len_.as<IntConstNode>()->val_;
                                if (len <= numSM) {
                                    l1 = mergedId;
                                } else if (len <= numSM * maxThreads) {
                                    std::tie(l1, l2) =
                                        split(mergedId, -1, numSM);
                                } else {
                                    std::tie(l1, l2) =
                                        split(mergedId, maxThreads);
                                }
                            } else {
                                // We don't use the `nparts` mode of `split`,
                                // because it will hinder dependence analysis.
                                // Instead, we use the `factor` mode and then
                                // reorder. See the doc string of `split` for
                                // details
                                std::tie(l2, l1) = split(mergedId, numSM);
                                reorder({l1, l2});
                                if (!findAll(l2).empty()) {
                                    std::tie(l1b, l2) = split(l2, maxThreads);
                                }
                            }
                        } else {
                            std::tie(l1, l2) = split(mergedId, maxThreads);
                        }
                        if (parallelizeAmongBlocks && l1.isValid() &&
                            !findAll(l1).empty()) {
                            if (l1b.isValid() && !findAll(l1b).empty()) {
                                // We are unable to fuse `l1` and `l1b` back to
                                // one loop. Because the length of `l1b` is not
                                // a constant, a division by this length will be
                                // introduced, which is not supported by ISL and
                                // may probably lead to false dependences
                                parallelize(l1, blockIdxY);
                                parallelize(l1b, blockIdxX);
                            } else {
                                parallelize(l1, blockIdxX);
                            }
                        }
                        if (l2.isValid() && !findAll(l2).empty()) {
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

            bool parallelizeInner = false;
            int64_t innerParallelSize = outerParallelSize;
            if (!done) {
                parallelizeInner = true;
            } else if (target->type() == TargetType::CPU) {
                if (mergedLen->nodeType() == ASTNodeType::IntConst) {
                    innerParallelSize *= mergedLen.as<IntConstNode>()->val_;
                    auto nCore = target.as<CPUTarget>()->nCore();
                    if (nCore.has_value() && *nCore / innerParallelSize > 1) {
                        parallelizeInner = true;
                    }
                }
            }
            if (parallelizeInner) {
                for (auto &&subLoop :
                     findAll("<For><-(!<For><-)*" + toString(root->id()))) {
                    autoParallelizeOuter(subLoop.as<ForNode>(), parentIsWarp,
                                         parallelizeAmongBlocks,
                                         innerParallelSize);
                }
            }
        };
    for (auto &&_root : findAll("<For><-(!<For><-)*<-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto root = _root.as<ForNode>();
        switch (target->type()) {
        case TargetType::CPU:
            autoParallelizeOuter(root, false, true, 1);
            break;

#ifdef FT_WITH_CUDA
        case TargetType::GPU:
            // Since there is no nested parallelism for GPU, we can only choose
            // one level to parallelize. If the outer most loop is too short, we
            // try the second outer loops instead
            if (auto &&inners =
                    findAll("<For><-(!<For><-)*" + toString(root->id()));
                inners.size() > 1 &&
                root->len_->nodeType() == ASTNodeType::IntConst &&
                root->len_.as<IntConstNode>()->val_ < 32) {
                for (auto &&inner : inners) {
                    autoParallelizeOuter(inner.as<ForNode>(), false, true, 1);
                }
            } else {
                autoParallelizeOuter(root, false, true, 1);
            }
            break;
#endif // FT_WITH_CUDA

        default:
            ASSERT(false);
        }
    }
}

} // namespace freetensor
