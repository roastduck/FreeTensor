#ifndef FREE_TENSOR_GPU_MAKE_SYNC_H
#define FREE_TENSOR_GPU_MAKE_SYNC_H

#ifdef FT_WITH_CUDA

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/find_loop_variance.h>
#include <driver/target.h>
#include <func.h>
#include <math/bounds.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

namespace gpu {

struct ThreadInfo {
    For loop_;
    bool inWarp_;
};

class FindAllThreads : public Visitor {
    int warpSize_;
    std::optional<int> thx_ = 1;
    std::optional<int> thy_ = 1;
    std::optional<int> thz_ = 1;
    std::unordered_map<ID, ThreadInfo> results_;

  public:
    FindAllThreads(const Ref<GPUTarget> &target)
        : warpSize_(target->warpSize()) {}

    const std::unordered_map<ID, ThreadInfo> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
};

class CopyParts : public Mutator {
    Expr cond_;
    const std::vector<Stmt> &splitters_;
    std::unordered_set<Stmt> fullParts_;

  public:
    CopyParts(const Expr &cond, const std::vector<Stmt> &splitters)
        : cond_(cond), splitters_(splitters) {}

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const StmtSeq &op) override;
};

struct CrossThreadDep {
    Stmt later_, earlier_, lcaStmt_, lcaLoop_;
    bool inWarp_;
    bool visiting_ = false, synced_ = false, syncedOnlyInBranch_ = false;

    CrossThreadDep(const Stmt &later, const Stmt &earlier, const Stmt &lcaStmt,
                   const Stmt &lcaLoop, bool inWarp)
        : later_(later), earlier_(earlier), lcaStmt_(lcaStmt),
          lcaLoop_(lcaLoop), inWarp_(inWarp) {}
};

class MakeSync : public Mutator {
    typedef Mutator BaseClass;

    Stmt root_;
    const std::unordered_map<ID, ThreadInfo> &loop2thread_;
    std::vector<CrossThreadDep> deps_;
    std::unordered_map<ID, std::pair<Stmt, bool /* isSyncWarp */>>
        syncBeforeFor_, syncBeforeIf_, syncBeforeLib_;
    std::unordered_map<ID, std::vector<Stmt>> branchSplittersThen_,
        branchSplittersElse_;
    LoopVariExprMap variantExprs_;

  public:
    MakeSync(const Stmt root,
             const std::unordered_map<ID, ThreadInfo> &loop2thread,
             std::vector<CrossThreadDep> &&deps, LoopVariExprMap &&variantExprs)
        : root_(root), loop2thread_(loop2thread), deps_(std::move(deps)),
          variantExprs_(std::move(variantExprs)) {}

  private:
    /**
     * Make `__syncthreads()` or `__syncwarp(__activemask())` statement.
     *
     * The semantics of the synchronization inserted intents to synchronize
     * among all threads RUNNING TO the synchronization point, not including the
     * threads in other control flows. For example,
     *
     * ```
     * if (threadIdx < X) {
     *   sync  // Sync point A
     * } else {
     *   sync  // Sync point B
     * }
     * ```
     *
     * Sync point A only synchronizes threads whose indices are < X. Sync point
     * B only syncrhonizes threads whose indices are >= X, while these two
     * groups of threads will not synchronize against each other.
     *
     * To correctly lower the code, the caller of these functions should decide
     * where to insert the synchronization (in this particular example, whether
     * to insert in the branch or out of it).
     *
     * Beside this consideration, CUDA does not allow `__syncthreads` in control
     * flow anyway. So `__syncthreads` should always be located out of branches.
     *
     * @{
     */
    static Stmt makeSyncThreads();
    static Stmt makeSyncWarp();
    /** @} */

    /**
     * Mark some `If` nodes to be split, in order to insert a synchronization
     *
     * When we are inserting a synchronization, like:
     *
     * ```
     * if cond
     *   ...
     *   sync
     *   ...
     *
     * ```
     *
     * ..., we can't insert it inside an `If` if `cond` is related to the
     * affected thread IDs. Otherwise, the threads entering the branch will wait
     * for the therads not entering the branch infinitely. We have to split the
     * `If`, like:
     *
     * ```
     * if cond
     *   ...
     * sync
     * if cond
     *   ...
     * ```
     *
     * However, when we need to add a synchronization inside a `For` which is
     * further inside an `If`, like:
     *
     * ```
     * if cond
     *   for ...
     *     ...
     *     sync
     *     ...
     * ```
     *
     * we need to move the `If` to outside of the `For` first. The final program
     * will be like:
     *
     * ```
     * for ...
     *   if cond
     *     ...
     *   sync
     *   if cond
     *     ...
     * ```
     *
     * Splitting an `If` is only needed when adding a `__syncthreads`, rather
     * than a `__syncwarp`, becasue the latter is simply a memory flush, not an
     * actual synchronization
     *
     * @param stmtInTree : Statement that we are inserting a synchronization
     * BESIDE. Must be in a tree with ancestors all the way to the root
     * @param sync : The new synchronization to insert
     * @param needSync : True if `__syncwarp`, false if `__syncthreads`
     */
    void markSyncForSplitting(const Stmt &stmtInTree, const Stmt &sync,
                              bool isSyncWarp);

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const MatMul &op) override;
};

Stmt makeSync(const Stmt &op, const Ref<GPUTarget> &target);

DEFINE_PASS_FOR_FUNC(makeSync)

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_MAKE_SYNC_H
