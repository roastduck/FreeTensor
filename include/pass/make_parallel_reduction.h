#ifndef MAKE_PARLLEL_REDUCTION_H
#define MAKE_PARLLEL_REDUCTION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/find_loop_variance.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

struct ParallelInfo {
    std::string type_;                    // parallel type
    std::vector<std::string> outerLoops_; // outer loop ID
};

class FindAllParallel : public Visitor {
    // Loop ID -> ParallelInfo
    std::unordered_map<std::string, ParallelInfo> results_;

    std::vector<std::string> loopStack_;

  public:
    const std::unordered_map<std::string, ParallelInfo> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
};

class MakeParallelReduction : public Mutator {
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        &toAlter_; // ReduceTo ID -> Racing For ID
    const LoopVariExprMap &variantMap_;

    std::unordered_map<std::string, std::string>
        paraScopes_; // For Id -> parallel
    std::unordered_map<std::string, std::vector<std::pair<ReduceOp, Expr>>>
        forReductions_;

  public:
    MakeParallelReduction(
        const std::unordered_map<std::string, std::unordered_set<std::string>>
            &toAlter,
        const LoopVariExprMap &variantMap)
        : toAlter_(toAlter), variantMap_(variantMap) {}

  protected:
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const For &op) override;
};

/**
 * Find all racing ReduceTo nodes, and implement them in a parallel way
 *
 * If all ReduceTo nodes in a parallel for all reduce into a loop-invariant
 * possition, we will use a race-free implementation. Otherwise, we will use
 * atomic operations.
 */
Stmt makeParallelReduction(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeParallelReduction)

} // namespace ir

#endif // MAKE_PARLLEL_REDUCTION_H
