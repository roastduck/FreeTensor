#ifndef FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H
#define FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
#include <analyze/find_loop_variance.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

struct ParallelInfo {
    ParallelScope type_;         // parallel type
    std::vector<ID> outerLoops_; // outer loop ID
};

class FindAllParallel : public Visitor {
    // Loop ID -> ParallelInfo
    std::unordered_map<ID, ParallelInfo> results_;

    std::vector<ID> loopStack_;

  public:
    const std::unordered_map<ID, ParallelInfo> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
};

class FindSerialLoopsOverReduce : public Visitor {
    std::unordered_map<ID, std::vector<For>>
        results_; // ReduceTo ID -> [For], from inner to outer
    std::vector<For> loopStack_;

  public:
    const std::unordered_map<ID, std::vector<For>> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const ReduceTo &op) override;
};

class MakeParallelReduction : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    struct ReductionItemFactors {
        ReduceOp op_;
        std::string var_;
        std::vector<std::vector<std::vector<Expr>>> lower_,
            upper_; // [dim][access][bound]
    };

    CompUniqueBounds unique_;

    const std::unordered_map<ID, std::unordered_set<ID>>
        &toAlter_; // ReduceTo ID -> Racing For ID
    const std::unordered_map<ID, std::vector<For>>
        &serialOverRed_; // ReduceTo ID -> [For], from inner to outer
    const LoopVariExprMap &variantMap_;

    std::unordered_map<ID, ParallelScope> paraScopes_; // For Id -> parallel
    std::unordered_map<ID, std::vector<ReductionItemFactors>> forReductions_;
    std::unordered_map<ID, std::unordered_set<std::string>>
        scopeDefined_; // For ID -> definitions at that scope
    std::unordered_map<
        ID,
        std::vector<std::tuple<ReduceTo, std::vector<Expr>, std::vector<Expr>>>>
        cacheAtomic_; // loop ID -> [(old ReduceTo node, new shape, new
                      // indices)]

  public:
    MakeParallelReduction(
        const std::unordered_map<ID, std::unordered_set<ID>> &toAlter,
        const std::unordered_map<ID, std::vector<For>> &serialOverRed,
        const LoopVariExprMap &variantMap)
        : unique_(*this), toAlter_(toAlter), serialOverRed_(serialOverRed),
          variantMap_(variantMap) {}

  protected:
    using BaseClass::visit;
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

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H
