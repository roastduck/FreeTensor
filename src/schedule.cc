#include <algorithm>

#include <analyze/all_defs.h>
#include <analyze/count_contig_access_loops.h>
#include <analyze/get_loop_nest_tree.h>
#include <pass/flatten_stmt_seq.h>
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
#include <schedule/parallelize.h>
#include <schedule/reorder.h>
#include <schedule/seperate_tail.h>
#include <schedule/set_mem_type.h>
#include <schedule/split.h>
#include <schedule/swap.h>
#include <schedule/unroll.h>
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

std::pair<std::string, std::string> Schedule::split(const std::string &id,
                                                    int factor, int nparts) {
    auto log = "split(" + id + ", factor=" + std::to_string(factor) +
               ", nparts=" + std::to_string(nparts) + ")";
    try {
        auto ret = ir::split(ast_, id, factor, nparts);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::reorder(const std::vector<std::string> &order) {
    std::string log = "reorder(";
    for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
        log += order[i] + (i < iEnd - 1 ? ", " : "");
    }
    log += ")";
    try {
        ast_ = ir::reorder(ast_, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(log + ": " + e.what());
    }
}

std::string Schedule::merge(const std::string &loop1,
                            const std::string &loop2) {
    auto log = "merge(" + loop1 + ", " + loop2 + ")";
    try {
        auto ret = ir::merge(ast_, loop1, loop2);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const std::string &loop, const std::string &after,
                  const std::string &suffix0, const std::string &suffix1) {
    auto log = "fission(" + loop + ", " + after + ")";
    try {
        auto ret = ir::fission(ast_, loop, after, suffix0, suffix1);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

std::string Schedule::fuse(const std::string &loop0, const std::string &loop1) {
    auto log = "fuse(" + loop0 + ", " + loop1 + ")";
    try {
        auto ret = ir::fuse(ast_, loop0, loop1);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::swap(const std::vector<std::string> &order) {
    std::string log = "swap(";
    for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
        log += order[i] + (i < iEnd - 1 ? ", " : "");
    }
    log += ")";
    try {
        ast_ = ir::swap(ast_, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::blend(const std::string &loop) {
    auto log = "blend(" + loop + ")";
    try {
        ast_ = ir::blend(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cache(const std::string &stmt, const std::string &var,
                MemType mtype) {
    auto log = "cache(" + stmt + ", " + var + ")";
    try {
        auto ret = ir::cache(ast_, stmt, var, mtype);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cacheReduction(const std::string &stmt, const std::string &var,
                         MemType mtype) {
    auto log = "cache_reduction(" + stmt + ", " + var + ")";
    try {
        auto ret = ir::cacheReduction(ast_, stmt, var, mtype);
        ast_ = ret.first;
        logs_.emplace_back(log);
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::setMemType(const std::string &def, MemType mtype) {
    auto log = "set_mem_type(" + def + ", " + toString(mtype) + ")";
    try {
        ast_ = ir::setMemType(ast_, def, mtype);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::varSplit(const std::string &def, int dim, VarSplitMode mode,
                        int factor, int nparts) {
    auto log =
        "var_split(" + def + ", " + std::to_string(dim) +
        (mode == VarSplitMode::FixedSize ? ", FixedSize" : ", RelaxedSize") +
        ", factor=" + std::to_string(factor) +
        ", nparts=" + std::to_string(nparts) + ")";
    try {
        ast_ = ir::varSplit(ast_, def, dim, mode, factor, nparts);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::varReorder(const std::string &def,
                          const std::vector<int> &order) {
    std::string log = "var_reorder(" + def + ", ";
    for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
        log += order[i] + (i < iEnd - 1 ? ", " : "");
    }
    log += ")";
    try {
        ast_ = ir::varReorder(ast_, def, order);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + "): " + e.what());
    }
}

std::string Schedule::moveTo(const std::string &_stmt, MoveToSide side,
                             const std::string &_dst) {
    auto bak = ast_;
    try {
        auto stmt = _stmt, dst = _dst;
        while (true) {
            ast_ = flattenStmtSeq(ast_, true);
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
                    std::vector<std::string> orderRev;
                    while (s.hasPrev() && movingUp()) {
                        s = s.prev();
                        orderRev.emplace_back(s.id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<std::string> order(orderRev.rbegin(),
                                                   orderRev.rend());
                    swap(order);
                } else {
                    while (!s.hasPrev() && movingUp()) {
                        s = s.outerCtrlFlow();
                    }
                    // TODO: Fission IfNode
                    ASSERT(s.node()->nodeType() == ASTNodeType::For);
                    // Leave IDs of the other statements unchanged
                    auto idMap = fission(s.id(), stmt, ".a", "").first;
                    stmt = idMap.at(s.id());
                }
                // TODO: Fuse if d is inner of s

            } else if (movingDown()) {
                if (s.hasNext()) {
                    std::vector<std::string> order;
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
                    // TODO: Fission IfNode
                    ASSERT(s.node()->nodeType() == ASTNodeType::For);
                    Cursor stmtCursor = getCursorById(ast_, stmt);
                    ASSERT(stmtCursor.hasPrev());
                    // Leave IDs of the other statements unchanged
                    auto idMap =
                        fission(s.id(), stmtCursor.prev().id(), "", ".b")
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
        throw InvalidSchedule("Invalid move_to(" + _stmt + ", " + _dst +
                              "): " + e.what());
    }
}

void Schedule::inlining(const std::string &def) {
    auto log = "inline(" + def + ")";
    try {
        ast_ = ir::inlining(ast_, def);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::parallelize(const std::string &loop,
                           const std::string &parallel) {
    auto log = "parallelize(" + loop + ", " + parallel + ")";
    try {
        ast_ = ir::parallelize(ast_, loop, parallel);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::unroll(const std::string &loop, bool immediate) {
    auto log = "unroll(" + loop + ")";
    try {
        ast_ = ir::unroll(ast_, loop, immediate);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::vectorize(const std::string &loop) {
    auto log = "vectorize(" + loop + ")";
    try {
        ast_ = ir::vectorize(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::seperateTail() { ast_ = ir::seperateTail(ast_); }

void Schedule::asMatMul(const std::string &loop) {
    auto log = "as_matmul(" + loop + ")";
    try {
        ast_ = ir::asMatMul(ast_, loop);
        logs_.emplace_back(log);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid " + log + ": " + e.what());
    }
}

void Schedule::autoSchedule(const Target &target) {
    autoFuse(target);
    autoParallelize(target);
    autoSetMemType(target);
}

void Schedule::autoFuse(const Target &target) {
    // Try to fuse each pair of consecutive loops
    std::function<void(const Ref<LoopNest> &nest)> visitNest =
        [&, this](const Ref<LoopNest> &nest) {
            Ref<LoopNest> last;
            std::string lastId;
            for (auto &&subNest : nest->subLoops_) {
                auto thisId = subNest->loop_->id();
                if (last.isValid()) {
                    auto bak = ast_;
                    auto logBak = logs_;
                    try {
                        // FIXME: This makes longformer slow
                        // lastId = moveTo(lastId, MoveToSide::Before, thisId);
                        thisId = fuse(lastId, thisId);
                        subNest->subLoops_.insert(subNest->subLoops_.begin(),
                                                  last->subLoops_.begin(),
                                                  last->subLoops_.end());
                        last->subLoops_.clear();
                    } catch (InvalidSchedule &e) {
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
        CountContigAccessLoops contigFinder;
        contigFinder(ast_);
        std::vector<std::pair<std::string, int>> contigLoops(
            contigFinder.counts().begin(), contigFinder.counts().end());
        std::sort(contigLoops.begin(), contigLoops.end(),
                  [](const std::pair<std::string, int> &lhs,
                     const std::pair<std::string, int> &rhs) {
                      return lhs.second > rhs.second;
                  });
        for (auto &&[loopId, cnt] : contigLoops) {
            auto loop = find(loopId);
            try {
                // FIXME: Do not hard-code 32
                auto [l0, l1] = split(loop.id(), 32);
                parallelize(l1, "threadIdx.x");

                // Reorder this scope to as outer as possible
                auto c = find(l1);
                if (c.hasOuter()) {
                    for (c = c.outer(); c.hasOuter(); c = c.outer()) {
                        if (c.nodeType() == ASTNodeType::For) {
                            try {
                                reorder({l1, c.id()});
                                l1 = c.id();
                            } catch (InvalidSchedule &e) {
                                break;
                            }
                        }
                    }
                }
            } catch (const InvalidSchedule &e) {
                // do nothing
            }
        }
    }

    // Try to merge and parallelize as many outer loops as possible
    auto loopNestTree = getLoopNestTree(ast_);
    for (const Ref<LoopNest> &root : loopNestTree->subLoops_) {
        auto latestSuccess = ast_;
        auto successLogs = logs_;

        try {
            Ref<LoopNest> loop = root;
            std::string loopId, outerId;
            while (true) {
                loopId = loop->loop_->id();
                if (!outerId.empty()) {
                    loopId = merge(outerId, loopId);
                }

                auto bak = ast_;
                auto logBak = logs_;
                switch (target.type()) {
                case TargetType::CPU:
                    parallelize(loopId, "openmp");
                    break;

                case TargetType::GPU: {
                    auto loop = find(loopId);
                    auto isParallelLoop = [](const Cursor &c) {
                        return c.nodeType() == ASTNodeType::For &&
                               !c.node()
                                    .as<ForNode>()
                                    ->property_.parallel_.empty();
                    };
                    bool childrenAllSerial =
                        getCursorByFilter(loop.node(), isParallelLoop).empty();
                    // 1. make sure all SMs are used
                    // 2. make sure blockDim is not too large
                    std::string l1, l2, l3;
                    // TODO: do not hard-code these numbers
                    std::tie(l1, l3) = split(loopId, -1, 80);
                    if (!findAll(l3).empty()) {
                        std::tie(l2, l3) =
                            split(l3, childrenAllSerial ? 256 : 8);
                    }
                    if (!findAll(l1).empty()) {
                        parallelize(l1, "blockIdx.y");
                    }
                    if (!findAll(l2).empty()) {
                        parallelize(l2, "blockIdx.x");
                    }
                    if (!findAll(l3).empty()) {
                        parallelize(l3, childrenAllSerial ? "threadIdx.x"
                                                          : "threadIdx.y");
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
    }
}

void Schedule::autoSetMemType(const Target &target) {
    // Try to put each VarDef as near to processor as possible
    if (target.type() == TargetType::GPU) {
        for (auto &&defId : allDefs(ast_, {AccessType::Cache})) {
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

} // namespace ir
