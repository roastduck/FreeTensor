#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
#include <schedule.h>
#include <schedule/check_loop_order.h>
#include <schedule/fission.h>
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
            std::find(dstOrder.begin(), dstOrder.end(), loop->id_) -
            dstOrder.begin());
    }

    // Bubble Sort
    size_t n = index.size();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j + 1 < n; j++) {
            if (index[j] > index[j + 1]) {
                findInvDeps(ast, {curOrder[j]->id_, curOrder[j + 1]->id_},
                            [&](const std::string &loop, const AST &later,
                                const AST &earlier) {
                                std::ostringstream os;
                                os << "Loop " << curOrder[j]->id_ << " and "
                                   << curOrder[j + 1]->id_
                                   << " are not permutable: Dependency "
                                   << (later->nodeType() == ASTNodeType::Load
                                           ? "READ "
                                           : "WRITE ")
                                   << later << " after "
                                   << (earlier->nodeType() == ASTNodeType::Load
                                           ? "READ "
                                           : "WRITE ")
                                   << earlier << " along loop " << loop
                                   << " cannot be resolved";
                                throw InvalidSchedule(std::regex_replace(
                                    os.str(), std::regex("\n"), ""));
                            });

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
    HoistVar hoist(loop, after);
    ast_ = hoist(ast_);
    FissionFor mutator(loop, after);
    ast_ = mutator(ast_);
    return std::make_pair(mutator.id0(), mutator.id1());
}

} // namespace ir

