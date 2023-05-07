#ifndef FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H
#define FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
#include <analyze/find_loop_variance.h>
#include <analyze/symbol_table.h>
#include <driver/target.h>
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

struct UseSyncInfo {
    // True if there is random access to the being-reduced variable. Note: even
    // there is no random access, we may also forced to use a synced reduction,
    // for example: cross-GPU-thread-block reduction can only be done by atomic.
    // In this case, `isRandomAccess_` is false
    bool isRandomAccess_;
};

/**
 * Lower supported parallel reductions to loop-carried reductions, which will be
 * further lowered to specific algorithms like binary reduction or local
 * accumulation in target-specific passes. Non-supported parallel reductions are
 * left for the following `MakeSyncReduction`.
 */
class MakeLoopCarriedReduction
    : public CompTransientBounds<SymbolTable<Mutator>> {
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
    const LoopVariExprMap &variantMap_;

    // ReduceTo ID -> whether randomly accessed. For all reductions in
    // `toAlter`, we first try to lower them as loop-carried reductions. If
    // impossible, we then insert them to this map, which is passed to
    // `MakeSyncReduction`.
    std::unordered_map<ID, UseSyncInfo> toUseSync_;

    std::unordered_map<ID, ParallelScope> paraScopes_; // For Id -> parallel
    std::unordered_map<ID, std::vector<ReductionItemFactors>> forReductions_;
    std::unordered_map<ID, std::unordered_set<std::string>>
        scopeDefined_; // For ID -> definitions at that scope

  public:
    MakeLoopCarriedReduction(
        const std::unordered_map<ID, std::unordered_set<ID>> &toAlter,
        const LoopVariExprMap &variantMap)
        : unique_(*this), toAlter_(toAlter), variantMap_(variantMap) {}

    const auto &toUseSync() const { return toUseSync_; }

  protected:
    using BaseClass::visit;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const For &op) override;
};

/**
 * Lower parallel reductions left by `MakeLoopCarriedReduction` to synchronized
 * reductions
 */
class MakeSyncReduction : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, UseSyncInfo> &toUseSync_;
    const std::unordered_map<ID, std::vector<For>>
        &serialOverRed_; // ReduceTo ID -> [For], from inner to outer
    const LoopVariExprMap &variantMap_;

#if defined(__GNUC__) && !defined(__clang__)
    // GCC<12 dose not support [[maybe_unused]] on member vars
    const Ref<Target> &target_;
#else
    [[maybe_unused]] /* used only if FT_WITH_CUDA */ const Ref<Target> &target_;
#endif

    struct SyncCacheInfo {
        ReduceTo oldNode_;
        std::vector<Expr> newShape_, newTargetIndices_;
        bool isRandomAccess_;
        std::vector<bool> preserveDim_;
    };
    std::unordered_map<ID,
                       std::vector<SyncCacheInfo>>
        cacheSync_; // loop ID -> [SyncCacheInfo]

    int64_t gpuThreadDim_ = 1;

  private:
    /**
     * Check whether we should put a `shape`-shaped variable to local memory on
     * a GPU
     *
     * Three types of shapes are rejected:
     *
     * 1. Large shapes, including:
     *   a) Shapes larger than the register count. Even we put them to local
     * memory, they will still land on DRAM. Since NVCC allocates stack frames
     * (where our local memory variables go to,
     * https://forums.developer.nvidia.com/t/out-of-memory-when-allocating-local-memory/238615)
     * by maximum possible thread count, so it will be worse than using global
     * memory.
     *   b) Large shapes that cannot be held in local memory for the stack-frame
     * reason above. Since we have no idea whether NVCC will put a local
     * variable in local memory or registers, in case NVCC does put it in local
     * memory, we don't want an OOM.
     * 2. Shapes with dynamic dimensions. Since indirect accessing of registers
     * is not supported, loops accessing a register array must be unrolled, but
     * dynamic dimensions make them impossible to unroll. Even we put these
     * variables to local memory, they will still land on DRAM. For the reason
     * mentioned in Case 1a, we prefer global memory over local memory.
     * 3. Variables being randomly accessed. Randomly access have to be
     * implemented via indirect access, even we can unroll the loops. Just like
     * Case 2, we put them to global memory.
     *
     * TODO: Move it to an architecture-specific pass
     */
    bool canResideInGPULocal(DataType dtype, const std::vector<Expr> &shape,
                             bool isRandomAccess) const;

    MemType localMType(MemType mtype, DataType dtype,
                       const std::vector<Expr> &shape,
                       bool isRandomAccess) const;

  public:
    MakeSyncReduction(
        const std::unordered_map<ID, UseSyncInfo> &toUseSync,
        const std::unordered_map<ID, std::vector<For>> &serialOverRed,
        const LoopVariExprMap &variantMap, const Ref<Target> &target)
        : toUseSync_(toUseSync), serialOverRed_(serialOverRed),
          variantMap_(variantMap), target_(target) {}

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
 * synchronized (including atomic) operations.
 *
 * @param target : Target information. Can be null for target-agnostic debugging
 */
Stmt makeParallelReduction(const Stmt &op, const Ref<Target> &target);

DEFINE_PASS_FOR_FUNC(makeParallelReduction)

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_PARLLEL_REDUCTION_H
