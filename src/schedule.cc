#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
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
#include <schedule/reorder.h>
#include <schedule/split.h>

namespace ir {

static std::string dep2Str(const std::string &loop, const std::string &var,
                           const AST &later, const AST &earlier) {
    std::ostringstream os;
    os << "Dependency "
       << (later->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ") << later
       << " after "
       << (earlier->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << earlier << " along loop " << loop << " cannot be resolved";
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
                    auto found =
                        [&](const std::vector<
                                std::pair<std::string, FindDepsMode>> &cond,
                            const std::string &var, const AST &later,
                            const AST &earlier) {
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
Schedule::fission(const std::string &loop, const std::string &after) {
    auto ast = ast_;
    HoistVar hoist(loop, after);
    FissionFor mutator(loop, after);
    try {
        ast = hoist(ast);
        if (!hoist.found()) {
            throw InvalidSchedule("Split point " + after +
                                  " not found inside " + loop);
        }

        auto &&xLoops = hoist.xLoops();

        // var name -> loop id
        std::vector<std::vector<std::pair<std::string, FindDepsMode>>> cond;
        for (const std::string &loop : hoist.innerLoops()) {
            cond.emplace_back(std::vector<std::pair<std::string, FindDepsMode>>{
                {loop, FindDepsMode::Normal},
                {hoist.seqId(), FindDepsMode::Inv}});
        }
        std::unordered_map<std::string, std::vector<std::string>> toAdd;
        auto found =
            [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
                const std::string &var, const AST &later, const AST &earlier) {
                ASSERT(cond.size() == 2);
                auto &&id = cond[0].first;
                if (!xLoops.count(var) ||
                    std::find(xLoops.at(var).begin(), xLoops.at(var).end(),
                              id) == xLoops.at(var).end()) {
                    throw InvalidSchedule(dep2Str(id, var, later, earlier));
                }
                if (std::find(toAdd[var].begin(), toAdd[var].end(), id) ==
                    toAdd[var].end()) {
                    toAdd[var].emplace_back(id);
                }
            };
        findDeps(ast, cond, found);

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
        auto found =
            [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
                const std::string &var, const AST &later, const AST &earlier) {
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

} // namespace ir

