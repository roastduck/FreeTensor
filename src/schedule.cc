#include <algorithm>

#include <itertools.hpp>

#include <analyze/all_defs.h>
#include <analyze/all_stmts.h>
#include <analyze/count_contig_access_loops.h>
#include <analyze/find_indexing_loops.h>
#include <analyze/get_loop_nest_tree.h>
#include <analyze/with_cursor.h>
#include <auto_schedule/utils.h>
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

namespace ir {

Schedule::Schedule(const Stmt &ast) : ast_(ast) { ast_ = simplifyPass(ast_); }

std::vector<Cursor>
Schedule::findAll(const std::function<bool(const Cursor &)> &filter) const {
    return getCursorByFilter(ast_, filter);
}

Cursor Schedule::find(const std::function<bool(const Cursor &)> &filter) const {
    auto ret = getCursorByFilter(ast_, filter);
    if (ret.size() != 1) {
        throw Error("find: There is " + std::to_string(ret.size()) +
                    " nodes matching the given condition. "
                    "Consider using findAll");
    }
    return ret[0];
}

std::pair<ID, ID> Schedule::split(const ID &id, int factor, int nparts) {
    auto log = "split(" + toString(id) + ", factor=" + std::to_string(factor) +
               ", nparts=" + std::to_string(nparts) + ")";
    try {
        auto ret = ir::split(ast_, id, factor, nparts);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::reorder(const std::vector<ID> &order) {
    std::string log = "reorder(";
    for (auto &&[i, item] : iter::enumerate(order)) {
        log += (i > 0 ? ", " : "") + toString(item);
    }
    log += ")";
    try {
        ast_ = ir::reorder(ast_, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log + ": " + e.what(), ast_);
    }
}

ID Schedule::merge(const ID &loop1, const ID &loop2) {
    auto log = "merge(" + toString(loop1) + ", " + toString(loop2) + ")";
    try {
        auto ret = ir::merge(ast_, loop1, loop2);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const ID &loop, FissionSide side, const ID &splitter,
                  const std::string &suffix0, const std::string &suffix1) {
    auto log = "fission(" + toString(loop) + ", " +
               (side == FissionSide::Before ? "BEFORE, " : "AFTER, ") +
               toString(splitter) + ")";
    try {
        auto ret = ir::fission(ast_, loop, side, splitter, suffix0, suffix1);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

ID Schedule::fuse(const ID &loop0, const ID &loop1, bool strict) {
    auto log = "fuse(" + toString(loop0) + ", " + toString(loop1) + ")";
    try {
        auto ret = ir::fuse(ast_, loop0, loop1, strict);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::swap(const std::vector<ID> &order) {
    std::string log = "swap(";
    for (auto &&[i, item] : iter::enumerate(order)) {
        log += (i > 0 ? ", " : "") + toString(item);
    }
    log += ")";
    try {
        ast_ = ir::swap(ast_, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::blend(const ID &loop) {
    auto log = "blend(" + toString(loop) + ")";
    try {
        ast_ = ir::blend(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cache(const ID &stmt, const std::string &var, MemType mtype) {
    auto log = "cache(" + toString(stmt) + ", " + var + ")";
    try {
        auto ret = ir::cache(ast_, stmt, var, mtype);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

std::tuple<ID, ID, std::string, ID>
Schedule::cacheReduction(const ID &stmt, const std::string &var,
                         MemType mtype) {
    auto log = "cache_reduction(" + toString(stmt) + ", " + var + ")";
    try {
        auto ret = ir::cacheReduction(ast_, stmt, var, mtype);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::setMemType(const ID &def, MemType mtype) {
    auto log = "set_mem_type(" + toString(def) + ", " + toString(mtype) + ")";
    try {
        ast_ = ir::setMemType(ast_, def, mtype);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::varSplit(const ID &def, int dim, VarSplitMode mode, int factor,
                        int nparts) {
    auto log =
        "var_split(" + toString(def) + ", " + std::to_string(dim) +
        (mode == VarSplitMode::FixedSize ? ", FixedSize" : ", RelaxedSize") +
        ", factor=" + std::to_string(factor) +
        ", nparts=" + std::to_string(nparts) + ")";
    try {
        ast_ = ir::varSplit(ast_, def, dim, mode, factor, nparts);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::varMerge(const ID &def, int dim) {
    auto log = "var_merge(" + toString(def) + ", " + std::to_string(dim) + ")";
    try {
        ast_ = ir::varMerge(ast_, def, dim);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::varReorder(const ID &def, const std::vector<int> &order) {
    std::string log = "var_reorder(" + toString(def) + ", ";
    for (auto &&[i, item] : iter::enumerate(order)) {
        log += (i > 0 ? ", " : "") + std::to_string(item);
    }
    log += ")";
    try {
        ast_ = ir::varReorder(ast_, def, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + "): " + e.what(), ast_);
    }
}

ID Schedule::moveTo(const ID &_stmt, MoveToSide side, const ID &_dst) {
    auto bak = ast_;
    try {
        auto stmt = _stmt, dst = _dst;
        while (true) {
            ast_ = hoistVarOverStmtSeq(ast_);
            Cursor s = getCursorById(ast_, stmt);
            Cursor d = getCursorById(ast_, dst);

            auto movingUp = [&]() {
                if (d.isOuter(s)) {
                    return side == MoveToSide::Before;
                }
                if (s.hasPrev()) {
                    return d.isBefore(side == MoveToSide::After ? s.prev() : s);
                } else {
                    return d.isBefore(s);
                }
            };
            auto movingDown = [&]() {
                if (d.isOuter(s)) {
                    return side == MoveToSide::After;
                }
                if (s.hasNext()) {
                    return (side == MoveToSide::Before ? s.next() : s)
                        .isBefore(d);
                } else {
                    return s.isBefore(d);
                }
            };

            if (movingUp()) {
                if (s.hasPrev()) {
                    std::vector<ID> orderRev;
                    while (s.hasPrev() && movingUp()) {
                        s = s.prev();
                        orderRev.emplace_back(s.id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<ID> order(orderRev.rbegin(), orderRev.rend());
                    swap(order);
                } else {
                    while (!s.hasPrev() && movingUp()) {
                        s = s.outerCtrlFlow();
                    }
                    if (s.node()->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            "Fission a If node in a StmtSeq is not currently "
                            "supported in moveTo",
                            ast_);
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMap =
                        fission(s.id(), FissionSide::After, stmt, ".a", "")
                            .first;
                    stmt = idMap.at(s.id());
                }
                // TODO: Fuse if d is inner of s

            } else if (movingDown()) {
                if (s.hasNext()) {
                    std::vector<ID> order;
                    while (s.hasNext() && movingDown()) {
                        s = s.next();
                        order.emplace_back(s.id());
                    }
                    order.emplace_back(stmt);
                    swap(order);
                } else {
                    while (!s.hasNext() && movingDown()) {
                        s = s.outerCtrlFlow();
                    }
                    if (s.node()->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            "Fission a If node in a StmtSeq is not currently "
                            "supported in moveTo",
                            ast_);
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMap =
                        fission(s.id(), FissionSide::Before, stmt, "", ".b")
                            .second;
                    stmt = idMap.at(s.id());
                }
                // TODO: Fuse if d is inner of s

            } else {
                return s.id();
            }
        }
    } catch (const InvalidSchedule &e) {
        ast_ = bak;
        throw InvalidSchedule("Invalid move_to(" + toString(_stmt) + ", " +
                                  toString(_dst) + "): " + e.what(),
                              ast_);
    }
}

void Schedule::inlining(const ID &def) {
    auto log = "inline(" + toString(def) + ")";
    try {
        ast_ = ir::inlining(ast_, def);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::parallelize(const ID &loop, const ParallelScope &parallel) {
    auto log =
        "parallelize(" + toString(loop) + ", " + toString(parallel) + ")";
    try {
        ast_ = ir::parallelize(ast_, loop, parallel);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::unroll(const ID &loop, bool immediate) {
    auto log = "unroll(" + toString(loop) + ")";
    try {
        ast_ = ir::unroll(ast_, loop, immediate);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::vectorize(const ID &loop) {
    auto log = "vectorize(" + toString(loop) + ")";
    try {
        ast_ = ir::vectorize(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::separateTail(bool noDuplicateVarDefs) {
    ast_ = ir::separateTail(ast_, noDuplicateVarDefs);
}

void Schedule::asMatMul(const ID &loop) {
    auto log = "as_matmul(" + toString(loop) + ")";
    try {
        ast_ = ir::asMatMul(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what(), ast_);
    }
}

void Schedule::autoSchedule(const Target &target) {
    autoUseLib(target);
    autoFuse(target);
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
                if (l->property_.preferLibs_) {
                    isPreferLibs = true;
                    break;
                }
                auto body = l->body_;
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

void Schedule::autoFuse(const Target &target) {
    // Try to fuse each pair of consecutive loops
    std::function<void(const Ref<LoopNest> &nest)> visitNest =
        [&, this](const Ref<LoopNest> &nest) {
            Ref<LoopNest> last;
            ID lastId;
            for (auto &&subNest : nest->subLoops_) {
                auto thisId = subNest->loop_->id();
                if (last.isValid()) {
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
            if (auto len = loop.node().as<ForNode>()->len_;
                len->nodeType() == ASTNodeType::IntConst &&
                len.as<IntConstNode>()->val_ <= 4) {
                continue;
            }

            auto bak = ast_;
            auto logBak = logs_;
            try {
                // FIXME: Do not hard-code 32
                auto [l0, l1] = split(loop.id(), 32);
                parallelize(l1, threadIdxX);

                try {
                    // Reorder this scope to as outer as possible
                    auto c = find(l1);
                    if (c.hasOuter()) {
                        for (c = c.outer(); c.hasOuter(); c = c.outer()) {
                            if (c.nodeType() == ASTNodeType::For) {
                                try {
                                    reorder({l1, c.id()});
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
                while (loop->loop_->property_.parallel_ != serialScope &&
                       loop->subLoops_.size() == 1) {
                    loop = loop->subLoops_.front();
                    parentIsWarp = true;
                }

                ID loopId, outerId;
                while (true) {
                    loopId = loop->loop_->id();
                    if (find(loopId)
                            .node()
                            .as<ForNode>()
                            ->property_.parallel_ != serialScope) {
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
                        auto isParallelLoop = [](const Cursor &c) {
                            return c.nodeType() == ASTNodeType::For &&
                                   c.node().as<ForNode>()
                                           ->property_.parallel_ != serialScope;
                        };
                        bool childIsWarp =
                            !getCursorByFilter(loop.node(), isParallelLoop)
                                 .empty();
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
                        if (auto loopNode = loop.node().as<ForNode>();
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
            if (loop->property_.parallel_ != serialScope ||
                loop->property_.vectorize_) {
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
                if (loop->property_.parallel_ == serialScope &&
                    !loop->property_.vectorize_ && !loop->property_.unroll_ &&
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

void Schedule::multiLevelTiling(const ForsWithDataReuse &target,
                                const MultiLevelTilingAnnotation &annotation,
                                const std::string &pat) {
    ir::multiLevelTiling(*this, target, annotation, pat);
}
void Schedule::multiLevelTilingWithFusion(
    const ForsWithDataReuse &target,
    const MultiLevelTilingAnnotation &annotation, const std::string &pat,
    const ElementWiseInfo &toFuse, int level) {
    ir::multiLevelTilingWithFusion(*this, target, annotation, pat, toFuse,
                                   level);
}
} // namespace ir
