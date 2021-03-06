#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/comp_access_bound.h>
#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <schedule.h>
#include <schedule/cache.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
#include <schedule/fuse.h>
#include <schedule/merge.h>
#include <schedule/parallelize.h>
#include <schedule/reorder.h>
#include <schedule/split.h>
#include <schedule/swap.h>
#include <schedule/unroll.h>

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

Cursor Schedule::find(const std::string &id) const {
    return getCursorById(ast_, id);
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
        ast = makeReduction(ast);

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
                    ast = prepareFindDeps(ast);

                    auto found = [&](const Dependency &d) {
                        ASSERT(d.cond_.size() == 1);
                        std::ostringstream os;
                        os << "Loop " << curOrder[j]->id() << " and "
                           << curOrder[j + 1]->id() << " are not permutable: "
                           << dep2Str(d.cond_[0].first, d.var_, d.later(),
                                      d.earlier());
                        throw InvalidSchedule(os.str());
                    };
                    findDeps(ast,
                             {{{curOrder[j]->id(), DepDirection::Inv}},
                              {{curOrder[j + 1]->id(), DepDirection::Inv}}},
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

        ast = prepareFindDeps(ast);

        auto variantExpr = findLoopVariance(ast);

        // var name -> loop id
        std::vector<std::vector<std::pair<std::string, DepDirection>>> disjunct;
        for (const std::string &inner : hoist.innerLoops()) {
            std::vector<std::pair<std::string, DepDirection>> conjunct{
                {inner, DepDirection::Normal},
                {hoist.seqId(), DepDirection::Inv}};
            disjunct.emplace_back(std::move(conjunct));
        }
        auto isRealWrite = [&](const std::string &loop, const AST &op) -> bool {
            if (op->nodeType() == ASTNodeType::Store) {
                auto expr = op.as<StoreNode>()->expr_;
                return variantExpr.count(expr) &&
                       variantExpr.at(expr).count(loop);
            } else if (op->nodeType() == ASTNodeType::ReduceTo) {
                auto expr = op.as<ReduceToNode>()->expr_;
                return variantExpr.count(expr) &&
                       variantExpr.at(expr).count(loop);
            } else {
                return false;
            }
        };
        std::unordered_map<std::string, std::vector<std::string>> toAdd;
        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() >= 2);
            auto &&id = d.cond_[d.cond_.size() - 2].first;
            if (!xLoops.count(d.var_) ||
                std::find(xLoops.at(d.var_).begin(), xLoops.at(d.var_).end(),
                          id) == xLoops.at(d.var_).end()) {
                throw InvalidSchedule(
                    dep2Str(id, d.var_, d.later(), d.earlier()));
            }
            if (!isRealWrite(id, d.later()) &&
                d.earlier()->nodeType() == ASTNodeType::Load) {
                return;
            }
            if (!isRealWrite(id, d.earlier()) &&
                d.later()->nodeType() == ASTNodeType::Load) {
                return;
            }
            if (std::find(toAdd[d.var_].begin(), toAdd[d.var_].end(), id) ==
                toAdd[d.var_].end()) {
                toAdd[d.var_].emplace_back(id);
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
        ast = prepareFindDeps(ast);

        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() == 1);
            throw InvalidSchedule(
                dep2Str(d.cond_[0].first, d.var_, d.later(), d.earlier()));
        };
        findDeps(ast, {{{loop0, DepDirection::Inv}}}, found);

        ast = mutator(ast);

        try {
            ast = simplifyPass(ast);
        } catch (const InvalidProgram &e) {
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
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            auto s0 = findParentStmt(later.cursor_);
            auto s1 = findParentStmt(earlier.cursor_);
            if (!s0.isValid() || !s1.isValid()) {
                return false;
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
            return (old0 < old1) != (new0 < new1);
        };
        auto found = [&](const Dependency &d) {
            throw InvalidSchedule(
                dep2Str(scope->id(), d.var_, d.later(), d.earlier()));
        };
        ast = prepareFindDeps(ast);
        findDeps(ast, {{{scope->id(), DepDirection::Normal}}}, found,
                 FindDepsMode::Dep, DEP_ALL, filter);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid reorder(";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
    ast_ = ast;
}

std::tuple<std::string, std::string, std::string>
Schedule::cache(const std::string &stmt, const std::string &var,
                MemType mtype) {
    auto ast = ast_;
    std::string fillStmt, flushStmt, newVar, newDef;
    try {
        MakeCacheVar makeCacheVar(stmt, var, mtype);
        ast = makeCacheVar(ast);
        newVar = makeCacheVar.newVar();
        newDef = makeCacheVar.newDef();
        if (newDef.empty()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }

        SimplifyPass::LowerBoundsMap lower;
        SimplifyPass::UpperBoundsMap upper;
        std::tie(ast, lower, upper) = simplifyAndGetBounds(ast);
        CompAccessBound compRBound(lower, upper, COMP_ACCESS_BOUND_READ);
        CompAccessBound compWBound(lower, upper, COMP_ACCESS_BOUND_WRITE);
        compRBound(ast);
        compWBound(ast);
        MakeFillAndFlush makeFillAndFlush(stmt, var, newVar, newDef,
                                          compRBound.results(),
                                          compWBound.results());
        ast = makeFillAndFlush(ast);
        fillStmt = makeFillAndFlush.fillStmt();
        flushStmt = makeFillAndFlush.flushStmt();

        ast = shrinkVar(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_tuple(std::move(fillStmt), std::move(flushStmt),
                           std::move(newVar));
}

std::tuple<std::string, std::string, std::string>
Schedule::cacheReduction(const std::string &stmt, const std::string &var,
                         MemType mtype) {
    auto ast = ast_;
    std::string initStmt, reduceStmt, newVar, newDef;
    try {
        ast = makeReduction(ast);

        MakeCacheVar makeCacheVar(stmt, var, mtype);
        ast = makeCacheVar(ast);
        newVar = makeCacheVar.newVar();
        newDef = makeCacheVar.newDef();
        if (newDef.empty()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }

        SimplifyPass::LowerBoundsMap lower;
        SimplifyPass::UpperBoundsMap upper;
        std::tie(ast, lower, upper) = simplifyAndGetBounds(ast);
        CompAccessBound compBound(lower, upper);
        compBound(ast);
        MakeInitAndReduce makeInitAndReduce(stmt, var, newVar, newDef,
                                            compBound.results());
        ast = makeInitAndReduce(ast);
        initStmt = makeInitAndReduce.initStmt();
        reduceStmt = makeInitAndReduce.reduceStmt();

        ast = shrinkVar(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache_reduction(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_tuple(std::move(initStmt), std::move(reduceStmt),
                           std::move(newVar));
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
                        s = s.outer();
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
                        s = s.outer();
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

void Schedule::unroll(const std::string &loop,
					  const unsigned int unroll_num) {
	auto ast = ast_;
	Unroll mutator(loop, unroll_num);
	try {
		ast = simplifyPass(mutator(ast));
		mutator.work = true;
        ast = mutator(ast);
        if (!mutator.done()) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid unroll(" + loop + ", " + std::to_string(unroll_num) +
                              "): " + e.what());
    }
    ast_ = ast;
}

} // namespace ir

