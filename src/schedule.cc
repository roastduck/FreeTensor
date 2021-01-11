#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
#include <analyze/disambiguous.h>
#include <analyze/find_loop_variance.h>
#include <cursor.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <schedule.h>
#include <schedule/cache_read.h>
#include <schedule/cache_write.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
#include <schedule/fuse.h>
#include <schedule/merge.h>
#include <schedule/parallelize.h>
#include <schedule/reorder.h>
#include <schedule/split.h>
#include <schedule/swap.h>

namespace ir {

static std::string dep2Str(const std::string &scope, const std::string &var,
                           const AST &later, const AST &earlier) {
    std::ostringstream os;
    os << "Dependency "
       << (later->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ") << later
       << " after "
       << (earlier->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << earlier << " along loop or statement block " << scope
       << " cannot be resolved";
    return std::regex_replace(os.str(), std::regex("\n"), "");
}

std::pair<std::string, std::string> Schedule::split(const std::string &id,
                                                    int factor, int nparts) {
    auto ast = ast_;
    Splitter mutator(id, factor, nparts);
    try {
        ast = mutator(ast);
        if (!mutator.found()) {
            throw InvalidSchedule("Loop not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(
            "Invalid split(" + id + ", factor=" + std::to_string(factor) +
            ", nparts=" + std::to_string(nparts) + "): " + e.what());
    }
    ast_ = simplifyPass(ast); // try to remove divisions, or it will hinder
                              // the dependency analysis
    return std::make_pair(mutator.outerId(), mutator.innerId());
}

void Schedule::reorder(const std::vector<std::string> &dstOrder) {
    auto ast = ast_;
    try {
        ast = MakeReduction()(ast);

        CheckLoopOrder checker(dstOrder);
        checker(ast);
        auto curOrder = checker.order();

        std::vector<int> index;
        index.reserve(curOrder.size());
        for (auto &&loop : curOrder) {
            index.emplace_back(
                std::find(dstOrder.begin(), dstOrder.end(), loop->id()) -
                dstOrder.begin());
        }

        // Bubble Sort
        size_t n = index.size();
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j + 1 < n; j++) {
                if (index[j] > index[j + 1]) {
                    ast = Disambiguous()(ast);

                    auto found =
                        [&](const std::vector<
                                std::pair<std::string, FindDepsMode>> &cond,
                            const std::string &var, const AST &later,
                            const AST &earlier, const Cursor &,
                            const Cursor &) {
                            ASSERT(cond.size() == 1);
                            std::ostringstream os;
                            os << "Loop " << curOrder[j]->id() << " and "
                               << curOrder[j + 1]->id()
                               << " are not permutable: "
                               << dep2Str(cond[0].first, var, later, earlier);
                            throw InvalidSchedule(os.str());
                        };
                    findDeps(ast,
                             {{{curOrder[j]->id(), FindDepsMode::Inv}},
                              {{curOrder[j + 1]->id(), FindDepsMode::Inv}}},
                             found);

                    SwapFor swapper(curOrder[j], curOrder[j + 1]);
                    ast = swapper(ast);
                    std::swap(index[j], index[j + 1]);
                    std::swap(curOrder[j], curOrder[j + 1]);
                }
            }
        }

    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid reorder(";
        for (size_t i = 0, iEnd = dstOrder.size(); i < iEnd; i++) {
            msg += dstOrder[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
    ast_ = ast;
}

std::string Schedule::merge(const std::string &loop1,
                            const std::string &loop2) {
    auto ast = ast_;
    std::string ret;
    try {
        CheckLoopOrder checker({loop1, loop2});
        checker(ast); // Check they are nested
        auto &&curOrder = checker.order();
        auto outer = curOrder[0], inner = curOrder[1];

        MergeFor mutator(outer, inner);
        ast = mutator(ast);
        ret = mutator.newIter();
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid merge(" + loop1 + ", " + loop2 +
                              "): " + e.what());
    }
    ast_ = ast;
    return ret;
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const std::string &loop, const std::string &after,
                  const std::string &suffix0, const std::string &suffix1) {
    if (suffix0 == suffix1) {
        throw InvalidSchedule(
            "fission: suffix0 cannot be the same with suffix1");
    }

    auto ast = ast_;
    HoistVar hoist(loop, after);
    FissionFor mutator(loop, after, suffix0, suffix1);
    try {
        ast = hoist(ast);
        if (!hoist.found()) {
            throw InvalidSchedule("Split point " + after +
                                  " not found inside " + loop);
        }
        auto &&xLoops = hoist.xLoops();

        ast = Disambiguous()(ast);

        auto variantExpr = findLoopVariance(ast);

        // var name -> loop id
        std::vector<std::vector<std::pair<std::string, FindDepsMode>>> disjunct;
        for (const std::string &inner : hoist.innerLoops()) {
            std::vector<std::pair<std::string, FindDepsMode>> conjunct{
                {inner, FindDepsMode::Normal},
                {hoist.seqId(), FindDepsMode::Inv}};
            disjunct.emplace_back(std::move(conjunct));
        }
        auto isRealWrite = [&](const std::string &loop, const AST &op) -> bool {
            if (op->nodeType() == ASTNodeType::Store) {
                auto expr = op.as<StoreNode>()->expr_;
                return variantExpr.count(expr.get()) &&
                       variantExpr.at(expr.get()).count(loop);
            } else if (op->nodeType() == ASTNodeType::AddTo) {
                auto expr = op.as<AddToNode>()->expr_;
                return variantExpr.count(expr.get()) &&
                       variantExpr.at(expr.get()).count(loop);
            } else {
                return false;
            }
        };
        std::unordered_map<std::string, std::vector<std::string>> toAdd;
        auto found =
            [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
                const std::string &var, const AST &later, const AST &earlier,
                const Cursor &, const Cursor &) {
                ASSERT(cond.size() >= 2);
                auto &&id = cond[cond.size() - 2].first;
                if (!xLoops.count(var) ||
                    std::find(xLoops.at(var).begin(), xLoops.at(var).end(),
                              id) == xLoops.at(var).end()) {
                    throw InvalidSchedule(dep2Str(id, var, later, earlier));
                }
                if (!isRealWrite(id, later) &&
                    earlier->nodeType() == ASTNodeType::Load) {
                    return;
                }
                if (!isRealWrite(id, earlier) &&
                    later->nodeType() == ASTNodeType::Load) {
                    return;
                }
                if (std::find(toAdd[var].begin(), toAdd[var].end(), id) ==
                    toAdd[var].end()) {
                    toAdd[var].emplace_back(id);
                }
            };
        findDeps(ast, disjunct, found);

        AddDimToVar adder(toAdd);
        ast = adder(ast);

        ast = mutator(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid fission(" + loop + ", " + after +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_pair(mutator.ids0(), mutator.ids1());
}

std::string Schedule::fuse(const std::string &loop0, const std::string &loop1) {
    auto ast = ast_;
    FuseFor mutator(loop0, loop1);
    try {
        ast = Disambiguous()(ast);

        auto found =
            [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
                const std::string &var, const AST &later, const AST &earlier,
                const Cursor &, const Cursor &) {
                ASSERT(cond.size() == 1);
                throw InvalidSchedule(
                    dep2Str(cond[0].first, var, later, earlier));
            };
        findDeps(ast, {{{loop0, FindDepsMode::Inv}}}, found);

        ast = mutator(ast);

        try {
            ast = simplifyPass(ast);
        } catch (const InvalidSchedule &e) {
            throw InvalidSchedule((std::string) "Fusing " + loop0 + " and " +
                                  loop1 + " loop1 with different lengths? " +
                                  e.what());
        }

        ast = shrinkVar(sinkVar(ast));
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid fuse(" + loop0 + ", " + loop1 +
                              "): " + e.what());
    }
    ast_ = ast;
    return mutator.fused();
}

void Schedule::swap(const std::vector<std::string> &order) {
    auto ast = ast_;
    try {
        Swap mutator(order);
        ast = mutator(ast);
        auto scope = mutator.scope();
        if (!scope.isValid()) {
            throw InvalidSchedule("Statements not found or not consecutive");
        }

        auto findParentStmt = [&](const Cursor &cursor) -> Stmt {
            for (auto &&item : order) {
                auto stmt = cursor.getParentById(item);
                if (stmt.isValid()) {
                    return stmt;
                }
            }
            return nullptr;
        };
        auto found =
            [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
                const std::string &var, const AST &later, const AST &earlier,
                const Cursor &laterCursor, const Cursor &earlierCursor) {
                auto s0 = findParentStmt(laterCursor);
                auto s1 = findParentStmt(earlierCursor);
                if (!s0.isValid() || !s1.isValid()) {
                    return;
                }
                auto old0 = std::find_if(
                    scope->stmts_.begin(), scope->stmts_.end(),
                    [&](const Stmt &s) { return s->id() == s0->id(); });
                auto old1 = std::find_if(
                    scope->stmts_.begin(), scope->stmts_.end(),
                    [&](const Stmt &s) { return s->id() == s1->id(); });
                auto new0 = std::find_if(
                    order.begin(), order.end(),
                    [&](const std::string &id) { return id == s0->id(); });
                auto new1 = std::find_if(
                    order.begin(), order.end(),
                    [&](const std::string &id) { return id == s1->id(); });
                if ((old0 < old1) != (new0 < new1)) {
                    throw InvalidSchedule(
                        dep2Str(scope->id(), var, later, earlier));
                }
            };
        ast = Disambiguous()(ast);
        findDeps(ast, {{{scope->id(), FindDepsMode::Normal}}}, found);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid reorder(";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
    ast_ = ast;
}

std::pair<std::string, std::string>
Schedule::cacheRead(const std::string &stmt, const std::string &var) {
    auto ast = ast_;
    CacheRead mutator(stmt, var);
    try {
        ast = mutator(ast);
        ast = shrinkVar(ast);
        if (!mutator.modified()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache_read(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_pair(mutator.fillStmt(), mutator.cacheVar());
}

std::pair<std::string, std::string>
Schedule::cacheWrite(const std::string &stmt, const std::string &var) {
    auto ast = ast_;
    CacheWrite mutator(stmt, var);
    try {
        ast = mutator(ast);
        ast = shrinkVar(ast);
        if (!mutator.modified()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache_write(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_pair(mutator.flushStmt(), mutator.cacheVar());
}

std::string Schedule::moveTo(const std::string &_stmt, const std::string &_dst,
                             bool toBegin, bool toEnd) {
    auto bak = ast_;
    try {
        auto stmt = _stmt, dst = _dst;
        while (true) {
            ast_ = flattenStmtSeq(ast_, true);
            Cursor s = getCursorById(ast_, stmt);
            Cursor d = getCursorById(ast_, dst);
            if (toBegin) {
                d.setMode(CursorMode::Begin);
            }
            if (toEnd) {
                d.setMode(CursorMode::End);
            }

            if (d.isBefore(s)) {
                if (s.hasPrev()) {
                    std::vector<std::string> orderRev;
                    if (!d.isBefore(s.prev())) {
                        return s.id();
                    }
                    while (s.hasPrev() && d.isBefore(s.prev())) {
                        s = s.prev();
                        orderRev.emplace_back(s.id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<std::string> order(orderRev.rbegin(),
                                                   orderRev.rend());
                    swap(order);
                } else {
                    if (!d.isBefore(s.outer())) {
                        return s.id();
                    }
                    while (!s.hasPrev() && d.isBefore(s.outer())) {
                        s = s.outer();
                    }
                    // TODO: Fission IfNode
                    ASSERT(s.top()->nodeType() == ASTNodeType::For);
                    // Leave IDs of the other statements unchanged
                    auto idMap = fission(s.id(), stmt, ".a", "").first;
                    stmt = idMap.at(s.id());
                }
                // TODO: Fuse if d is inner of s
            } else {
                // TODO
                ASSERT(false);
            }
        }
    } catch (const InvalidSchedule &e) {
        ast_ = bak;
        throw InvalidSchedule("Invalid move_to(" + _stmt + ", " + _dst +
                              "): " + e.what());
    }
}

void Schedule::parallelize(const std::string &loop,
                           const std::string &parallel) {
    auto ast = ast_;
    Parallelize mutator(loop, parallel);
    try {
        ast = mutator(ast);
        if (!mutator.done()) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid parallelize(" + loop + ", " + parallel +
                              "): " + e.what());
    }
    ast_ = ast;
}

} // namespace ir

