#ifndef FREE_TENSOR_SCHEDULE_H
#define FREE_TENSOR_SCHEDULE_H

#include <functional>
#include <unordered_map>

#include <auto_schedule/structs.h>
#include <driver/target.h>
#include <func.h>
#include <schedule/fission.h>
#include <schedule/memoized_schedules.h>
#include <schedule/schedule_log.h>
#include <schedule/var_split.h>
#include <stmt.h>

namespace freetensor {

enum class MoveToSide : int { Before, After };

class Schedule {
    Func func_;
    Stmt ast_;

    int verbose_ = 0;

    ScheduleLog logs_;
    Ref<MemoizedSchedules> memoized_;

  private:
    void saveSuccessLog(const ScheduleLog &logs);

  public:
    Schedule() = default;
    Schedule(const Stmt &ast, int verbose = 0);
    Schedule(const Func &func, int verbose = 0)
        : Schedule(func->body_, verbose) {
        func_ = func;
    }

    /**
     * Copy the `Schedule` object for trying different scheduling decisions in
     * the future
     *
     * The `fork`ed object shares the same `MemoizedSchedule` with the original
     * one, so common decisions can be saved and reused
     */
    Schedule fork() const { return *this; }

    /**
     * @return : The function being transformed
     */
    Func func() const {
        ASSERT(func_.isValid());
        return makeFunc(func_->name_, func_->params_, func_->returns_, ast_);
    }

    /**
     * @return : The statements being transformed, without a function signature
     */
    Stmt ast() const;

    /**
     * @return : Logs of all schedules applied
     */
    std::vector<Ref<ScheduleLogItem>> logs() const;

    /**
     * Find all nodes in the current AST satisfying a given condition
     *
     * @param filter : A callback. Return true for acceptance
     */
    std::vector<Stmt>
    findAll(const std::function<bool(const Stmt &)> &filter) const;

    /**
     * Find the only one nodes in the current AST satisfying a given condition
     *
     * @param filter : A callback. Return true for acceptance
     * @throw Error : if there is more than one, or there is no node found
     */
    Stmt find(const std::function<bool(const Stmt &)> &filter) const;

    /**
     * Find a (maybe non-existing) node in the current AST by ID
     *
     * @param id: ID
     */
    std::vector<Stmt> findAll(const ID &id) const {
        return findAll([&id](const Stmt &c) { return c->id() == id; });
    }

    /**
     * Find a node in the current AST by ID
     *
     * @param id: ID
     */
    Stmt find(const ID &id) const {
        return find([&id](const Stmt &c) { return c->id() == id; });
    }

    /**
     * Split a loop into two nested loops
     *
     * To fission a loop into two consecutive loops, use `fission` instead
     *
     * Two modes are provided:
     *
     * 1. Specify `factor` and leave `nparts` to -1. It will result in an outer
     * loop with length `ceil(n / factor)`, and an inner loop with length
     * `factor`, where `n` is the original loop length added by `shift`. The
     * original iterator `i` will be transformed to `i0 * factor + i1`, where
     * `i0` and `i1` are the iterators of the new outer and inner loops,
     * respectively
     * 2. Specify `nparts` and leave `factor` to -1. It will result in an
     * outer loop with length `nparts`, and an inner loop with length `ceil(n /
     * nparts)`, where `n` is the original loop length added by `shift`. The
     * original iterator `i` will be transformed to `i0 * ceil(n / nparts) +
     * i1`, where `i0` and `i1` are the iterators of the new outer and inner
     * loops, respectively
     *
     * Please note that the second mode will introduce an `i0 * ceil(n /
     * nparts)` factor into the program, which cannot be recognized by
     * polyhedral analysis, which may hinder some following schedules. If
     * possible, please use the first mode, and then reorder the inner and outer
     * loops
     *
     * @param id : ID of the loop to be split
     * @param factor : Length of the inner loop. Set to -1 if using `nparts`
     * @param nparts : Length of the outer loop. Set to -1 if using `factor`
     * @param shift : Shift of iteration base. Defaults to zero
     * @throw InvalidSchedule if the loop is not found
     * @return : (outer loop ID, inner loop ID)
     */
    std::pair<ID, ID> split(const ID &id, int factor = -1, int nparts = -1,
                            int shift = 0);

    /**
     * Reorder directly nested loops
     *
     * To swap consecutive loops, use `swap` instead
     *
     * @param order : Vector of loop IDs. The requested order of the loops
     * @throw InvalidSchedule if the input is invalid or there are breaking
     * dependencies
     */
    void reorder(const std::vector<ID> &order);

    /**
     * Merge two directly nested loops into one
     *
     * To fuse consecutive loops, use `fuse` instead
     *
     * `parallelize`, `unroll` and `vectorize` properties will be reset on the
     * merged loop
     *
     * @param loop1, loop2 : ID of the loops to be merged, can be in any order
     * @throw InvalidSchedule if the loops are not directly nested
     * @return : ID of the merged loop
     */
    ID merge(const ID &loop1, const ID &loop2);

    typedef std::unordered_map<ID, ID> IDMap;
    /**
     * Fission a loop into two loops each containing part of the statements, one
     * followed by another
     *
     * To split loop into two nested loops, use `split` instead
     *
     * @param loop : ID of the loop to be fissioned
     * @param side : If `After`, `splitter` is the last statement of the first
     * loop. If `Before`, `splitter` is the first statement of the second loop
     * @param splitter : Where to fission the loop
     * @param suffix0 : ID suffix of the statements in the first loop, default
     * to ".a", can be "" for convenience, but cannot be the same with suffix1
     * @param suffix1 : ID suffix of the statements in the second loop, default
     * to ".b", can be "" for convenience, but cannot be the same with suffix0
     * @throw InvalidSchedule if any dependency cannot be resolved
     * @return : ({old ID -> new ID in 1st loop}, {old ID -> new ID in 2nd
     * loop})
     */
    std::pair<IDMap, IDMap> fission(const ID &loop, FissionSide side,
                                    const ID &splitter,
                                    const std::string &suffix0 = ".a",
                                    const std::string &suffix1 = ".b");

    /**
     * Fuse two directly following loops with the same length into one
     *
     * To merge nested loops into one, use `merge` instead
     *
     * `parallelize`, `unroll` and `vectorize` properties will be reset on the
     * fused loop
     *
     * @param loop0 : ID of the leading loop
     * @param loop1 : ID of the following loop. If omitted, it will try to find
     * a following loop of `loop0`
     * @param strict : If true, throw an error if unable to determine whether
     * the two loops are of the same length
     * @throw InvalidSchedule if the two loops are not directly following, the
     * two loops are not of the same length, or there is any dependency cannot
     * be resolved
     * @return : ID of the result loop
     * @{
     */
    ID fuse(const ID &loop0, const ID &loop1, bool strict = false);
    ID fuse(const ID &loop0, bool strict = false);
    /** @} */

    /**
     * Swap statements in the same block
     *
     * To reorder nested loops, use `reorder` instead
     *
     * @param order : list of IDs of the statements
     * @throw InvalidSchedule if the statements are not found or the
     * dependencies cannot be solved
     */
    void swap(const std::vector<ID> &order);

    /**
     * Unroll a loop and interleave statements from each iteration
     *
     * E.g.
     *
     * ```
     * for i = 0 to 2 {
     *   f(i);
     *   g(i);
     * }
     * ```
     *
     * will be transformed to be
     *
     * ```
     * f(0);
     * f(1);
     * g(0);
     * g(1);
     * ```
     *
     * Virtual threads in TVM can be implemented via blend
     *
     * @param loop : ID of the loop being transformed
     * @throw InvalidSchedule if the loop is not found, the loop length is not a
     * constant, or the dependencies cannot be solved
     */
    void blend(const ID &loop);

    /**
     * Cache a variable into a new local variable
     *
     * All needed data will be filled into the cache first, then all reads and
     * writes will be directed to the cache, and finally all needed data will be
     * flushed from the cache
     *
     * Note for reduction: This transformation preserves the computation order.
     * It will transform
     *
     * ```
     * a += x
     * a += y
     * ```
     *
     * to
     *
     * ```
     * a.cache = a + x + y
     * a = a.cache
     * ```
     *
     * If you need a "real" cache for reduction, which reorders the computation,
     * use `cache_reduction` instead
     *
     * @param stmt : ID of the statement or block (e.g. an If or a For) to be
     * modified
     * @param var : name of the variable to be cached
     * @param mtype : where to cache
     * @throw InvalidSchedule if the ID or name is not found
     * @return : (ID of the statement that fills the cache, ID of the statement
     * that flushes from the cache, name of the cache variable, ID of the VarDef
     * node of the cache variable)
     */
    std::tuple<ID, ID, std::string, ID>
    cache(const ID &stmt, const std::string &var, MemType mtype);

    /**
     * Perform local reductions (e.g. sum) in a local variable first, and then
     * reduce the local result to the global variable
     *
     * E.g.
     *
     * ```
     * a += x
     * a += y
     * ```
     *
     * will be transformed to be
     *
     * ```
     * a.cache = x + y
     * a += a.cache
     * ```
     *
     * @param stmt : ID of the statement or block (e.g. an If or a For) to be
     * modified
     * @param var : name of the variable to be cached. Only reductions are
     * allowed on `var` in `stmt`. Plain reads or writes are not allowed
     * @param mtype : where to cache
     * @throw InvalidSchedule if the ID or name is not found, or there are
     * unsupported reads or writes
     * @return : (ID of the statement that initialize the cache, ID of the
     * statement that reduces the local result to the global result, name of the
     * cache variable, ID of the VarDef node of the cache variable)
     */
    std::tuple<ID, ID, std::string, ID>
    cacheReduction(const ID &stmt, const std::string &var, MemType mtype);

    /**
     * Change where a variable is stored
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param mtype : Where the variable should be stored
     * @throw InvalidSchedule if the variable is not found
     */
    void setMemType(const ID &def, MemType mtype);

    /**
     * Split a dimension of a variable into two
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param dim : which dimension to be split
     * @param mode : When the dimension to split is not divisible by `factor` or
     * `nparts`, the resulting shape may become larger. In `FixedSize` mode, the
     * actual buffer size will not be changed, and gurads will be added to
     * prevent out-of-bound accesses. In `RelaxedSize` mode, the buffer size may
     * increase. The `RelaxedSize` mode cannot be applied to I/O variables
     * @param factor : Length of the inner (higher no.) dimension. Set to -1 if
     * using `nparts`
     * @param nparts : Length of the outer (lower no.) loop. Set to -1 if using
     * `factor`
     * @throw InvalidSchedule if the variable or the dimension is not found
     */
    void varSplit(const ID &def, int dim, VarSplitMode mode, int factor = -1,
                  int nparts = -1);

    /**
     * Merge two dimensions of a variable
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param dim : Merge the `dim`-th and the `(dim + 1)`-th dimension
     */
    void varMerge(const ID &def, int dim);

    /**
     * Reorder the dimensions of a variable
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param order : new order of the dimensions
     * @throw InvalidSchedule if the variable or the order is illegal
     */
    void varReorder(const ID &def, const std::vector<int> &order);

    /**
     * Move a statement to a new position
     *
     * This is a composite schedule command, which is implemented with other
     * commands
     *
     * @param stmt : ID of the statement to be moved
     * @param side : Whether `stmt` will be BEFORE or AFTER `dst
     * @param dst : Insert `stmt` to be directly after this statement
     * @throw InvalidSchedule if there is no feasible path to move
     * @return : The new ID of stmt
     */
    ID moveTo(const ID &stmt, MoveToSide side, const ID &dst);

    /**
     * Remove a variable. When the variable is used, recompute its value
     *
     * @param def : ID of the VarDef statement of the specific variable. It can
     * not be an I/O varible
     * @throw InvalidSchedule if the variable cannot be completely removed
     */
    void inlining(const ID &def);

    /**
     * Mark a loop with a parallel implementation
     *
     * @param loop : ID of the loop
     * @param parallel : Parallel scope
     */
    void parallelize(const ID &loop, const ParallelScope &parallel);

    /**
     * Unroll a loop
     *
     * @param loop : ID of the loop
     * @param immediate : If false (by default), postpone the unroll procedure
     * to the backend compiler, which saves scheduling time. If true, unroll the
     * loop immediately, which may help further simplifications based on the
     * unrolled result. If your purpose is just to fill the instruction cache,
     * set it to false. If you are unrolling a loop that computes array indices,
     * set it to true
     * @throw InvalidSchedule if the loop is not found or length of the loop is
     * not a constant
     */
    void unroll(const ID &loop, bool immediate = false);

    /**
     * Vectorize a loop
     *
     * Please note that, as vectorization is different from architecture to
     * achitecture, the scheduler may or may not postpone it to the backend
     * compiler. The vectorization is a best-effort schedule
     *
     * @param loop : ID of the loop
     * @throw InvalidSchedule if the ID or name is not found, or the dependency
     * requirement is not met
     */
    void vectorize(const ID &loop);

    /**
     * Seperate main iterations and tail iterations of a loop
     *
     * E.g.
     *
     * ```
     * for i = 0 -> 3 {
     *   for j = 0 -> 4 {
     *      if (i * 4 + j < 10) {
     *        ...
     *      }
     *   }
     * }
     * ```
     *
     * Each loop will be separated into 2 parts: the body and the tail. After
     * simplification, the program will finally be transformed to
     *
     * ```
     * for i = 0 -> 2 {
     *   for j = 0 -> 4 {
     *     ...
     *   }
     * }
     * for j = 0 -> 2 {
     *   ...
     * }
     * ```
     *
     * Ideally, all programs can benefit from this schedule. However, this
     * schedule may greatly increase the program size and make the compiling
     * time way too long. Therefore, this transformation is implemented as a
     * schedule, which can be applied optionally. (TODO: Optionally apply this
     * schedule to part of the program)
     *
     * @param noDuplicateVarDefs : If there is two VarDef nodes in two branches,
     * it may result in doubled memory use, since different thread may go to
     * different branch. Set this parameter to true to stop duplicating VarDef
     * nodes.
     */
    void separateTail(bool noDuplicateVarDefs = false);

    /**
     * Transform nested loops to be a external call to a matrix multiplication
     *
     * @param loop: ID of the loop
     * @throw InvalidSchedule if the loop cannot be transformed to be a matrix
     * multiplication
     */
    void asMatMul(const ID &loop);

    /**
     * (Experimental) Automatic scheduling using some heuristics
     *
     * @param target : Target architecture
     */
    void autoSchedule(const Target &target);

    /**
     * (Experimental) Automatically use external libs using some heuristics
     *
     * @param target : Target architecture
     */
    void autoUseLib(const Target &target);

    /**
     * (Experimental) Automatically fuse consecutive loops using some heuristics
     *
     * @param target : Target architecture
     */
    void autoFuse(const Target &target);

    /**
     * (Experimental) Automatically parallelize some loops using some heuristics
     *
     * @param target : Target architecture
     */
    void autoParallelize(const Target &target);

    /**
     * (Experimental) Automatically set memory types using some heuristics
     *
     * @param target : Target architecture
     */
    void autoSetMemType(const Target &target);

    /**
     * (Experimental) Automatically unroll loops using some heuristics
     *
     * @param target : Target architecture
     */
    void autoUnroll(const Target &target);

    std::vector<std::pair<ID, int>>
    multiLevelTiling(const ForsWithDataReuse &target,
                     const MultiLevelTilingAnnotation &annotation,
                     const std::string &pat, int level);

    std::vector<std::pair<ID, int>>
    multiLevelTilingWithFusion(const ForsWithDataReuse &target,
                               const MultiLevelTilingAnnotation &annotation,
                               const std::string &pat,
                               const ElementWiseInfo &toFuse, int level,
                               TargetType targetType, bool doCacheRead);
};

} // namespace freetensor

#endif // FREE_TENSOR_SCHEDULE_H
