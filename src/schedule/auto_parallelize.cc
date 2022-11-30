#include <analyze/count_contig_access_loops.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoParallelize(const Ref<Target> &target) {
#ifdef FT_WITH_CUDA
    // [GPU only] Try to parallelize loops accessing contiguous items as warps
    if (target->type() == TargetType::GPU) {
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
                    split(loop->id(), target.as<GPUTarget>()->warpSize());
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

                switch (target->type()) {
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
                    // save time for dependence analysis. If not, we split
                    // it twice, and merge once
                    int numSM = target.as<GPUTarget>()->multiProcessorCount();
                    int maxThreads = 256; // Can be max thread per block (1024),
                                          // but our generated kernels are huge,
                                          // so set it lower to reserve for more
                                          // registers. TODO: no magic number
                    if (parentIsWarp || childIsWarp) {
                        maxThreads /= target.as<GPUTarget>()->warpSize();
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
                    if (!findAll(l1).empty()) {
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
    for (auto &&_root : findAll("<For><-(!<For><-)*<-|")) {
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

} // namespace freetensor
