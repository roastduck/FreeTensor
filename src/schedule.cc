#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/comp_access_bound.h>
#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/remove_writes.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/z3_simplify.h>
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
#include <schedule/var_split.h>
#include <schedule/vectorize.h>

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
                    auto filter = [&](const AccessPoint &later,
                                      const AccessPoint &earlier) {
                        return earlier.cursor_
                                   .getParentById(curOrder[j + 1]->id())
                                   .isValid() &&
                               later.cursor_
                                   .getParentById(curOrder[j + 1]->id())
                                   .isValid();
                    };
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
                             found, FindDepsMode::Dep, DEP_ALL, filter);

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
        ret = mutator.newId();
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
    // FIXME: Check the condition is not variant when splitting an If

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

        auto variantExpr = findLoopVariance(ast);

        // var name -> loop id
        std::vector<std::vector<std::pair<std::string, DepDirection>>> disjunct;
        for (const std::string &inner : hoist.innerLoops()) {
            disjunct.push_back({{inner, DepDirection::Normal}});
        }
        auto isRealWrite = [&](const std::string &loop,
                               const VarDef &def) -> bool {
            return isVariant(variantExpr.second, def, loop);
        };
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(hoist.afterId()).isValid() &&
                   later.cursor_.getParentById(hoist.beforeId()).isValid();
        };
        std::unordered_map<std::string, std::vector<std::string>> toAdd;
        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() == 1);
            auto &&id = d.cond_[0].first;
            if (!xLoops.count(d.var_) ||
                std::find(xLoops.at(d.var_).begin(), xLoops.at(d.var_).end(),
                          id) == xLoops.at(d.var_).end()) {
                throw InvalidSchedule(
                    dep2Str(id, d.var_, d.later(), d.earlier()));
            }
            if (!isRealWrite(id, d.def()) &&
                d.earlier()->nodeType() == ASTNodeType::Load) {
                return;
            }
            if (!isRealWrite(id, d.def()) &&
                d.later()->nodeType() == ASTNodeType::Load) {
                return;
            }
            if (std::find(toAdd[d.defId()].begin(), toAdd[d.defId()].end(),
                          id) == toAdd[d.defId()].end()) {
                toAdd[d.defId()].emplace_back(id);
            }
        };
        findDeps(ast, disjunct, found, FindDepsMode::Dep, DEP_ALL, filter);

        AddDimToVar adder(toAdd);
        ast = adder(ast);

        ast = mutator(ast);
        ast = mergeAndHoistIf(ast);
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
    CheckAccessible check(loop0, loop1);
    try {
        check(ast);
        if (!check.loop0().loop_.isValid()) {
            throw InvalidSchedule("Loops not found in a StmtSeq");
        }

        for (auto &&stmt : check.loop1().surroundings_) {
            if (stmt->nodeType() == ASTNodeType::VarDef) {
                for (auto shape :
                     stmt.as<VarDefNode>()->buffer_->tensor().shape()) {
                    if (!checkNotModified(
                            ast, shape, CheckNotModifiedSide::Before, loop0,
                            CheckNotModifiedSide::Before, loop1)) {
                        throw InvalidSchedule((
                            std::string) "The shape of Vars in loop1 shouldn't "
                                         "be changed in loop0");
                    }
                }
            }
        }

        ast = mutator(ast);

        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(mutator.afterId()).isValid() &&
                   later.cursor_.getParentById(mutator.beforeId()).isValid();
        };
        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() == 1);
            throw InvalidSchedule(
                dep2Str(d.cond_[0].first, d.var_, d.later(), d.earlier()));
        };
        findDeps(ast, {{{mutator.fused(), DepDirection::Normal}}}, found,
                 FindDepsMode::Dep, DEP_ALL, filter);

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
        findDeps(ast, {{{scope->id(), DepDirection::Normal}}}, found,
                 FindDepsMode::Dep, DEP_ALL, filter);
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid swap(";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::blend(const std::string &loop) {
    auto ast = ast_;
    try {
        ast = simplifyPass(ast); // Const prop for ForNode::len_

        FindAllScopesInside finder(loop);
        finder(ast);
        if (!finder.found()) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }

        std::vector<std::vector<std::pair<std::string, DepDirection>>> cond;
        for (auto &&item : finder.scopes()) {
            cond.push_back(
                {{loop, DepDirection::Normal}, {item, DepDirection::Inv}});
        }
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(loop).isValid() &&
                   later.cursor_.getParentById(loop).isValid();
        };
        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() == 2);
            throw InvalidSchedule(
                dep2Str(d.cond_[1].first, d.var_, d.later(), d.earlier()));
        };
        findDeps(ast, cond, found, FindDepsMode::Dep, DEP_ALL, filter);

        auto loopVari = findLoopVariance(ast);
        ast = BlendPass(loop, loopVari.first, loopVari.second)(ast);
        ast = flattenStmtSeq(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid blend(" + loop + "): " + e.what());
    }
    ast_ = ast;
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cache(const std::string &stmt, const std::string &var,
                MemType mtype) {
    auto ast = ast_;
    std::string fillStmt, flushStmt, newVar, oldDef, newDef;
    try {
        MakeCacheVar makeCacheVar(stmt, var, mtype, false);
        ast = makeCacheVar(ast);
        newVar = makeCacheVar.newVar();
        oldDef = makeCacheVar.oldDef();
        newDef = makeCacheVar.newDef();
        if (newDef.empty()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }

        BuiltinSimplify::LowerBoundsMap lower;
        BuiltinSimplify::UpperBoundsMap upper;
        std::tie(ast, lower, upper) =
            simplifyAndGetBounds<BuiltinSimplify>(ast);
        auto rBound =
            compAccessBound(ast, newDef, lower, upper, COMP_ACCESS_BOUND_READ);
        auto wBound =
            compAccessBound(ast, newDef, lower, upper, COMP_ACCESS_BOUND_WRITE);
        MakeFillAndFlush makeFillAndFlush(stmt, var, newVar, oldDef, rBound,
                                          wBound);
        ast = makeFillAndFlush(ast);
        fillStmt = makeFillAndFlush.fillStmt();
        flushStmt = makeFillAndFlush.flushStmt();

        ast = shrinkSingleVar(ast, newDef);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_tuple(std::move(fillStmt), std::move(flushStmt),
                           std::move(newVar), std::move(newDef));
}

std::tuple<std::string, std::string, std::string, std::string>
Schedule::cacheReduction(const std::string &stmt, const std::string &var,
                         MemType mtype) {
    auto ast = ast_;
    std::string initStmt, reduceStmt, newVar, oldDef, newDef;
    try {
        ast = makeReduction(ast);

        MakeCacheVar makeCacheVar(stmt, var, mtype, true);
        ast = makeCacheVar(ast);
        newVar = makeCacheVar.newVar();
        oldDef = makeCacheVar.oldDef();
        newDef = makeCacheVar.newDef();
        if (newDef.empty()) {
            throw InvalidSchedule("Statement " + stmt + " not found");
        }

        BuiltinSimplify::LowerBoundsMap lower;
        BuiltinSimplify::UpperBoundsMap upper;
        std::tie(ast, lower, upper) =
            simplifyAndGetBounds<BuiltinSimplify>(ast);
        auto bound = compAccessBound(ast, newDef, lower, upper);
        MakeInitAndReduce makeInitAndReduce(stmt, var, newVar, oldDef, newDef,
                                            bound);
        ast = makeInitAndReduce(ast);
        initStmt = makeInitAndReduce.initStmt();
        reduceStmt = makeInitAndReduce.reduceStmt();

        ast = shrinkSingleVar(ast, newDef);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid cache_reduction(" + stmt + ", " + var +
                              "): " + e.what());
    }
    ast_ = ast;
    return std::make_tuple(std::move(initStmt), std::move(reduceStmt),
                           std::move(newVar), std::move(newDef));
}

void Schedule::setMemType(const std::string &def, MemType mtype) {
    auto ast = ast_;
    try {
        SetMemType mutator(def, mtype);
        ast = mutator(ast);
        if (!mutator.found()) {
            throw InvalidSchedule(def + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid set_mtype(" + def + ", " +
                              toString(mtype) + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::varSplit(const std::string &def, int dim, VarSplitMode mode,
                        int factor, int nparts) {
    auto ast = ast_;
    try {
        VarSplit mutator(def, dim, mode == VarSplitMode::FixedSize, factor,
                         nparts);
        ast = mutator(ast);
        if (!mutator.found()) {
            throw InvalidSchedule(def + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule(
            "Invalid var_split(" + def + ", " + std::to_string(dim) +
            (mode == VarSplitMode::FixedSize ? ", FixedSize"
                                             : ", RelaxedSize") +
            ", factor=" + std::to_string(factor) +
            ", nparts=" + std::to_string(nparts) + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::varReorder(const std::string &def,
                          const std::vector<int> &order) {
    auto ast = ast_;
    try {
        VarReorder mutator(def, order);
        ast = mutator(ast);
        if (!mutator.found()) {
            throw InvalidSchedule(def + " not found");
        }
    } catch (const InvalidSchedule &e) {
        std::string msg = "Invalid var_reorder(" + def + ", ";
        for (size_t i = 0, iEnd = order.size(); i < iEnd; i++) {
            msg += order[i] + (i < iEnd - 1 ? ", " : "");
        }
        throw InvalidSchedule(msg + "): " + e.what());
    }
    ast_ = ast;
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
    auto ast = ast_;
    try {
        std::unordered_map<Load, Expr> replace;
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return later.op_->nodeType() == ASTNodeType::Load &&
                   earlier.def_->id() == def;
        };
        auto found = [&](const Dependency &dep) {
            if (replace.count(dep.later().as<LoadNode>())) {
                throw InvalidSchedule("Multiple writes correspond to one read");
            }
            Expr expr;
            if (dep.earlier()->nodeType() == ASTNodeType::Store) {
                auto earlier = dep.earlier().as<StoreNode>();
                expr = MakeInlinePlaceholder(earlier->indices_)(earlier->expr_);
            } else {
                ASSERT(dep.earlier()->nodeType() == ASTNodeType::ReduceTo);
                auto earlier = dep.earlier().as<ReduceToNode>();
                expr = MakeInlinePlaceholder(earlier->indices_)(earlier->expr_);
            }
            if (!checkNotModified(ast, expr, CheckNotModifiedSide::After,
                                  dep.earlier_.cursor_.id(),
                                  CheckNotModifiedSide::Before,
                                  dep.later_.cursor_.id())) {
                throw InvalidSchedule(
                    "The expression will be modified after inlining from " +
                    toString(dep.earlier_.cursor_.node()) + " into " +
                    toString(dep.later_.cursor_.node()));
            }
            auto later = dep.later().as<LoadNode>();
            replace[later] = ApplyInlinePlaceholder(later->indices_)(expr);
        };
        findDeps(ast, {{}}, found, FindDepsMode::KillLater, DEP_RAW, filter);
        ast = MakeInline(def, replace)(ast);
        ast = simplifyPass(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid inline(" + def + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::parallelize(const std::string &loop,
                           const std::string &parallel) {
    auto ast = ast_;
    Parallelize mutator(loop, parallel);
    try {
        ast = makeReduction(ast);
        auto oldAst = ast;
        ast = mutator(ast);
        if (!mutator.done()) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }
        FindDepsCond findDepsCond{{loop, DepDirection::Normal}};
        for (auto &&outerLoop : mutator.outerLoops()) {
            findDepsCond.push_back({outerLoop, DepDirection::Same});
        }
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(loop).isValid() &&
                   later.cursor_.getParentById(loop).isValid();
        };
        auto found = [&](const Dependency &d) {
            throw InvalidSchedule(
                dep2Str(loop, d.var_, d.later(), d.earlier()));
        };
        findDeps(oldAst, {findDepsCond}, found, FindDepsMode::Dep, DEP_ALL,
                 filter);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid parallelize(" + loop + ", " + parallel +
                              "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::unroll(const std::string &loop, bool immediate) {
    auto ast = ast_;
    try {
        ast = simplifyPass(ast); // Const prop for ForNode::len_
        bool done = false;
        if (immediate) {
            ImmediateUnroll mutator(loop);
            ast = mutator(ast);
            done = mutator.done();
            ast = removeWrites(ast);
        } else {
            BackUnroll mutator(loop);
            ast = mutator(ast);
            done = mutator.done();
        }
        if (!done) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid unroll(" + loop + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::vectorize(const std::string &loop) {
    auto ast = ast_;
    Vectorize mutator(loop);
    try {
        ast = mutator(ast);
        if (!mutator.done()) {
            throw InvalidSchedule("Loop " + loop + " not found");
        }
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(loop).isValid() &&
                   later.cursor_.getParentById(loop).isValid();
        };
        auto found = [&](const Dependency &d) {
            throw InvalidSchedule(
                dep2Str(loop, d.var_, d.later(), d.earlier()));
        };
        findDeps(ast, {{{loop, DepDirection::Normal}}}, found,
                 FindDepsMode::Dep, DEP_ALL, filter);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid vectorize(" + loop + "): " + e.what());
    }
    ast_ = ast;
}

void Schedule::seperateTail() {
    auto ast = ast_;

    FindAllIfs finder;
    finder(ast);
    auto candidates = finder.results();

    while (!candidates.empty()) {
        SeperateTail mutator(candidates);
        ast = mutator(ast);
        ast =
            z3Simplify(ast); // Although Z3 may be slow, if we don't use Z3
                             // here, there will be too many redundant branches,
                             // which will make each pass even slower
        candidates = mutator.nextCandidates();
    }

    ast_ = ast;
}

void Schedule::asMatMul(const std::string &loop) {
    auto ast = ast_;
    try {
        ast = simplifyPass(ast); // const prop
        ast = makeReduction(ast);
        ast = AsMatMul(loop)(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule("Invalid as_matmul(" + loop + "): " + e.what());
    }
    ast_ = ast;
}

} // namespace ir
