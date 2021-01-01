#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
#include <pass/simplify.h>
#include <schedule.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
#include <schedule/fuse.h>
#include <schedule/merge.h>
#include <schedule/reorder.h>
#include <schedule/split.h>

namespace ir {

std::pair<std::string, std::string> Schedule::split(const std::string &id,
                                                    int factor, int nparts) {
    Splitter mutator(id, factor, nparts);
    ast_ = mutator(ast_);
    return std::make_pair(mutator.outerId(), mutator.innerId());
}

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

void Schedule::reorder(const std::vector<std::string> &dstOrder) {
    auto ast = ast_;

    // BEGIN: MAY THROW: Don't use ast_
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
                    [&](const std::vector<std::pair<std::string, FindDepsMode>>
                            &cond,
                        const std::string &var, const AST &later,
                        const AST &earlier) {
                        ASSERT(cond.size() == 1);
                        std::ostringstream os;
                        os << "Loop " << curOrder[j]->id() << " and "
                           << curOrder[j + 1]->id() << " are not permutable: "
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

    // END: MAY THROW
    ast_ = ast;
}

std::string Schedule::merge(const std::string &loop1,
                            const std::string &loop2) {
    CheckLoopOrder checker({loop1, loop2});
    checker(ast_); // Check they are nested
    auto &&curOrder = checker.order();
    auto outer = curOrder[0], inner = curOrder[1];

    MergeFor mutator(outer, inner);
    ast_ = mutator(ast_);
    return mutator.newIter();
}

std::pair<std::string, std::string>
Schedule::fission(const std::string &loop, const std::string &after) {
    auto ast = ast_;

    // BEGIN: MAY THROW: Don't use ast_
    HoistVar hoist(loop, after);
    ast = hoist(ast);
    auto &&xLoops = hoist.xLoops();

    // var name -> loop id
    std::vector<std::vector<std::pair<std::string, FindDepsMode>>> cond;
    for (const std::string &loop : hoist.innerLoops()) {
        cond.emplace_back(std::vector<std::pair<std::string, FindDepsMode>>{
            {loop, FindDepsMode::Normal}, {hoist.seqId(), FindDepsMode::Inv}});
    }
    std::unordered_map<std::string, std::vector<std::string>> toAdd;
    auto found =
        [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
            const std::string &var, const AST &later, const AST &earlier) {
            ASSERT(cond.size() == 2);
            auto &&id = cond[0].first;
            if (!xLoops.count(var) ||
                std::find(xLoops.at(var).begin(), xLoops.at(var).end(), id) ==
                    xLoops.at(var).end()) {
                throw InvalidSchedule("Unable to fission: " +
                                      dep2Str(id, var, later, earlier));
            }
            if (std::find(toAdd[var].begin(), toAdd[var].end(), id) ==
                toAdd[var].end()) {
                toAdd[var].emplace_back(id);
            }
        };
    findDeps(ast, cond, found);

    AddDimToVar adder(toAdd);
    ast = adder(ast);

    FissionFor mutator(loop, after);
    ast = mutator(ast);

    // END: MAY THROW
    ast_ = ast;
    return std::make_pair(mutator.id0(), mutator.id1());
}

std::string Schedule::fuse(const std::string &loop0, const std::string &loop1) {
    auto ast = ast_;

    // BEGIN: MAY THROW: Don't use ast_
    auto found =
        [&](const std::vector<std::pair<std::string, FindDepsMode>> &cond,
            const std::string &var, const AST &later, const AST &earlier) {
            ASSERT(cond.size() == 1);
            std::ostringstream os;
            os << "Unable to fuse " << loop0 << " and " << loop1 << ": "
               << dep2Str(cond[0].first, var, later, earlier);
            throw InvalidSchedule(os.str());
        };
    findDeps(ast, {{{loop0, FindDepsMode::Inv}}}, found);

    FuseFor mutator(loop0, loop1);
    ast = mutator(ast);

    try {
        ast = simplifyPass(ast);
    } catch (const InvalidSchedule &e) {
        throw InvalidSchedule((std::string) "Fusing " + loop0 + " and " +
                              loop1 + " loop1 with different lengths? " +
                              e.what());
    }

    // END: MAY THROW
    ast_ = ast;
    return mutator.fused();
}

} // namespace ir

