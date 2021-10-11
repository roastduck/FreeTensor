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
    try {
        auto ret = ir::split(ast_, id, factor, nparts);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(
            "Invalid split(" + id + ", factor=" + std::to_string(factor) +
            ", nparts=" + std::to_string(nparts) + "): " + e.what());
    }
}

void Schedule::reorder(const std::vector<std::string> &order) {
    try {
        ast_ = ir::reorder(ast_, order);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid reorder(";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
}

std::string Schedule::merge(const std::string &loop1,
                            const std::string &loop2) {
    try {
        auto ret = ir::merge(ast_, loop1, loop2);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid merge(" + loop1 + ", " + loop2 +
                              "): " + e.what());
    }
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const std::string &loop, const std::string &after,
                  const std::string &suffix0, const std::string &suffix1) {
    try {
        auto ret = ir::fission(ast_, loop, after, suffix0, suffix1);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid fission(" + loop + ", " + after +
                              "): " + e.what());
    }
}

std::string Schedule::fuse(const std::string &loop0, const std::string &loop1) {
    try {
        auto ret = ir::fuse(ast_, loop0, loop1);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid fuse(" + loop0 + ", " + loop1 +
                              "): " + e.what());
    }
}

void Schedule::swap(const std::vector<std::string> &order) {
    try {
        ast_ = ir::swap(ast_, order);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid swap(";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
}

void Schedule::blend(const std::string &loop) {
    try {
        ast_ = ir::blend(ast_, loop);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid blend(" + loop + "): " + e.what());
    }
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cache(const std::string &stmt, const std::string &var,
                MemType mtype) {
    try {
        auto ret = ir::cache(ast_, stmt, var, mtype);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache(" + stmt + ", " + var +
                              "): " + e.what());
    }
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cacheReduction(const std::string &stmt, const std::string &var,
                         MemType mtype) {
    try {
        auto ret = ir::cacheReduction(ast_, stmt, var, mtype);
        ast_ = ret.first;
        return ret.second;
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache_reduction(" + stmt + ", " + var +
                              "): " + e.what());
    }
}

void Schedule::setMemType(const std::string &def, MemType mtype) {
    try {
        ast_ = ir::setMemType(ast_, def, mtype);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid set_mtype(" + def + ", " +
                              toString(mtype) + "): " + e.what());
    }
}

void Schedule::varSplit(const std::string &def, int dim, VarSplitMode mode,
                        int factor, int nparts) {
    try {
        ast_ = ir::varSplit(ast_, def, dim, mode, factor, nparts);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(
            "Invalid var_split(" + def + ", " + std::to_string(dim) +
            (mode == VarSplitMode::FixedSize ? ", FixedSize"
                                             : ", RelaxedSize") +
            ", factor=" + std::to_string(factor) +
            ", nparts=" + std::to_string(nparts) + "): " + e.what());
    }
}

void Schedule::varReorder(const std::string &def,
                          const std::vector<int> &order) {
    try {
        ast_ = ir::varReorder(ast_, def, order);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid var_reorder(" + def + ", ";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
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
    try {
        ast_ = ir::inlining(ast_, def);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid inline(" + def + "): " + e.what());
    }
}

void Schedule::parallelize(const std::string &loop,
                           const std::string &parallel) {
    try {
        ast_ = ir::parallelize(ast_, loop, parallel);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid parallelize(" + loop + ", " + parallel +
                              "): " + e.what());
    }
}

void Schedule::unroll(const std::string &loop, bool immediate) {
    try {
        ast_ = ir::unroll(ast_, loop, immediate);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid unroll(" + loop + "): " + e.what());
    }
}

void Schedule::vectorize(const std::string &loop) {
    try {
        ast_ = ir::vectorize(ast_, loop);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid vectorize(" + loop + "): " + e.what());
    }
}

void Schedule::seperateTail() { ast_ = ir::seperateTail(ast_); }

void Schedule::asMatMul(const std::string &loop) {
    try {
        ast_ = ir::asMatMul(ast_, loop);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid as_matmul(" + loop + "): " + e.what());
    }
}

} // namespace ir
