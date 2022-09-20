#ifndef FREE_TENSOR_SCHEDULE_H
#define FREE_TENSOR_SCHEDULE_H

#include <functional>
#include <unordered_map>

#include <analyze/find_stmt.h>
#include <auto_schedule/structs.h>
#include <driver/target.h>
#include <func.h>
#include <probability/rand_ctx.h>
#include <random.h>
#include <schedule/fission.h>
#include <schedule/memoized_schedules.h>
#include <schedule/schedule_log.h>
#include <schedule/var_split.h>
#include <stmt.h>

namespace freetensor {

enum class MoveToSide : int { Before, After };

struct AutoScheduleTuneTrial {
    Ref<RandTrace> trace_;
    Func lowered_;
    std::string code_;
    double time_, stddev_;
};

class Schedule {
    struct Transaction {
        Stmt ast_;
        ScheduleLog logs_;

        Transaction(const Stmt &ast, const ScheduleLog &logs)
            : ast_(ast), logs_(logs) {}
    };

    Func func_; /// Used for `func()`. Only header of `func_` is used, while its
                /// body is `ast_` in `openTrans_`

    std::vector<Transaction> openTrans_; /// Open transactions

    int verbose_ = 0;

    Ref<MemoizedSchedules> memoized_;

    Ref<OpenMPRandomEngine> rng_;
    Ref<RandCtx<OpenMPRandomEngine>> randCtx_;

  private:
    void setAst(const Stmt &ast);
    void setLogs(const ScheduleLog &log);

    /**
     * Prelude and conclude passes for each schedule
     *
     * Some quick passes that performed when the `Schedule` object is
     * constructed and after each schedule
     *
     * These passes shall not change the statment IDs and Metadata, which means
     * return values from each schedule will be preserved
     */
    static Stmt quickOptimizations(const Stmt &ast);

    /**
     * Generate a function that invokes a schedule on the current `ast()`, and
     * apply `quickOptimizations`
     */
    template <class F> auto futureSchedule(const F &sched) {
        return [&](auto &&...args) {
            auto ret = sched(ast(), std::forward<decltype(args)>(args)...);
            if constexpr (std::convertible_to<decltype(ret), Stmt>) {
                return quickOptimizations(ret);
            } else { // pair(Stmt, other info)
                return std::make_pair(quickOptimizations(ret.first),
                                      ret.second);
            }
        };
    }

    /**
     * Append a new schedule log to logs, and try looking up an identical
     * schedule from `MemoizedSchedules`
     *
     * If a memoized log is found, the memoized schedule result (including
     * exceptions, if any) can be reused. If not found, save the new log to
     * `MemoziedSchedules`
     */
    template <class T> T appendLog(const T &_log) {
        auto log = _log;
        setLogs(memoized_->lookupOrCreate(logs().push(log)));
        ASSERT(logs().top()->type() == log->type());
        log = logs().top().as<typename decltype(log)::Object>();
        log->run();
        return log;
    }

    /**
     * Apply a schedule log
     *
     * If the log is memoized, simply retrieve the memoized result. If the
     * result is an exception, re-throw it
     *
     * `Schedule::ast()` is updated, and other return values are returned
     */
    template <class T> auto applyLog(const T &log) {
        auto ret = log->getResult();
        if constexpr (std::convertible_to<decltype(ret), Stmt>) {
            setAst(ret);
            return;
        } else { // pair(Stmt, other info)
            setAst(ret.first);
            return ret.second;
        }
    }

  public:
    Schedule() = default;
    Schedule(const Stmt &ast, int verbose = 0);
    Schedule(const Func &func, int verbose = 0)
        : Schedule(func->body_, verbose) {
        func_ = func;
    }

    // Copy by default, which means `Ref`s in a `Schedule` object is shared
    Schedule(const Schedule &) = default;
    Schedule &operator=(const Schedule &) = default;

    /**
     * Copy the `Schedule` object for trying different scheduling decisions in
     * the future
     *
     * The `fork`ed object shares the same `MemoizedSchedule` with the original
     * one, so common decisions can be saved and reused
     *
     * The `fork`ed object shares the same `RandCtx` objects, so it can learn
     * from multiple scheduling trials
     */
    Schedule fork() const { return *this; }

    /**
     * Transaction of schedules
     *
     * Schedules are applied in transactions. A transaction is created with
     * `beginTransaction()`, applied as a whole with `commitTransaction()`, and
     * can be aborted with `abortTransaction()`
     *
     * Transactions can be nested. Technically, each schedule is by itself a
     * inner-most transaction, while a `Schedule` object defines the outer-most
     * transaction, but these inner-most and outer-most transcations are
     * invisible to users
     *
     * @{
     */
    void beginTransaction();
    void commitTransaction();
    void abortTransaction();
    /** @} */

    /**
     * @return : The function being transformed
     */
    Func func() const {
        ASSERT(func_.isValid());
        return makeFunc(func_->name_, func_->params_, func_->returns_, ast());
    }

    /**
     * @return : The statements being transformed, without a function signature
     */
    const Stmt &ast() const;

    /**
     * @return : Logs of all schedules applied
     */
    const ScheduleLog &logs() const;

    /**
     * Verbose level
     */
    int verbose() const { return verbose_; }

    /**
     * Find all nodes (maybe non-existing) in the current AST satisfying a given
     * condition
     *
     * @param filter : A callback that returns true for acceptance, or a
     * `Selector`, or an `ID`
     */
    template <class T> std::vector<Stmt> findAll(const T &filter) const {
        return findAllStmt(ast(), filter);
    }

    /**
     * Find all nodes (at least one) in the current AST satisfying a given
     * condition
     *
     * @param filter : A callback that returns true for acceptance, or a
     * `Selector`, or an `ID`
     */
    template <class T> std::vector<Stmt> findAtLeastOne(const T &filter) const {
        auto ret = findAllStmt(ast(), filter);
        if (ret.empty()) {
            throw InvalidSchedule(ast(), "No statement found by filter");
        }
        return ret;
    }

    /**
     * Find the only one nodes in the current AST satisfying a given condition
     *
     * @param filter : A callback that returns true for acceptance, or a
     * `Selector`, or an `ID`
     * @throw InvalidSchedule : if there is more than one, or there is no node
     * found
     */
    template <class T> Stmt find(const T &filter) const {
        try {
            return findStmt(ast(), filter);
        } catch (const UnexpectedQueryResult &e) {
            throw InvalidSchedule(ast(), e.what());
        }
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
     * Suppose the original loop is labeled "L", the split two loops can be
     * selected by "$split.0{L}" (the outer loop) and "$split.1{L}" (the inner
     * loop). If one of the resulting loop is proved to have only a single
     * iteration, it will be removed
     *
     * @param id : ID of the loop to be split
     * @param factor : Length of the inner loop. Set to -1 if using `nparts`
     * @param nparts : Length of the outer loop. Set to -1 if using `factor`
     * @param shift : Shift of iteration base. Defaults to zero
     * @throw InvalidSchedule if the loop is not found
     * @return : (outer loop ID, inner loop ID), either ID can be invalid if
     * the loop is proved to have only a single iteration
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
     * Suppose the original loops are labeled "L1" and "L2", the merged loop can
     * be selected by "$merge{L1, L2}"
     *
     * @param loop1, loop2 : ID of the loops to be merged, can be in any order
     * @throw InvalidSchedule if the loops are not directly nested
     * @return : ID of the merged loop
     */
    ID merge(const ID &loop1, const ID &loop2);

    /**
     * Permute perfectly nested loops (directly nested loops without statements
     * in between) with the given loop space transformation function
     *
     * The transformed loops follow ascending lexical order of the transformed
     * terms returned by `transformFunc` when called with original iteration
     * variables
     *
     * @param loopsId : the list of IDs of perfectly nested loops to be permuted
     * @param transformFunc : the loop space transformation function, should be
     * bijective
     * @throw InvalidSchedule if the loops are not perfectly nested, or the
     * permutation is not bijective, or the permutation breaks certain
     * dependency
     * @return : the list of IDs of permuted loops
     */
    std::vector<ID>
    permute(const std::vector<ID> &loopsId,
            const std::function<std::vector<Expr>(std::vector<Expr>)>
                &transformFunc);

    typedef std::unordered_map<ID, ID> IDMap;
    /**
     * Fission a loop into two loops each containing part of the statements, one
     * followed by another
     *
     * To split loop into two nested loops, use `split` instead
     *
     * Statements inside the original loop will be distributed to one or both
     * (happening if they are scope statements) loops. If a statement is
     * originally labeled "S", it can be selected by "$fission.0{S}" (from the
     * first loop) or "$fission.1{S}" (from the second loop) after fission. If
     * one of the resulting loop has an empty body, it will be removed
     *
     * @param loop : ID of the loop to be fissioned
     * @param side : If `After`, `splitter` is the last statement of the first
     * loop. If `Before`, `splitter` is the first statement of the second loop
     * @param splitter : Where to fission the loop
     * @param suffix0 : The suffix in the `op` of metadata of result part 0. If
     * empty, the fissioned part 0 preserves original ID and metadata. Cannot be
     * empty together with `suffix1`.
     * @param suffix1 : The suffix in the `op` of metadata of result part 1. If
     * empty, the fissioned part 1 preserves original ID and metadata. Cannot be
     * empty together with `suffix0`.
     * @throw InvalidSchedule if any dependency cannot be resolved
     * @return : ({old ID -> new ID in 1st loop}, {old ID -> new ID in 2nd
     * loop}). If a loop is removed because it has an empty body, it will not be
     * in the returned map
     */
    std::pair<IDMap, IDMap> fission(const ID &loop, FissionSide side,
                                    const ID &splitter,
                                    const std::string &suffix0 = ".0",
                                    const std::string &suffix1 = ".1");

    /**
     * Fuse two directly following loops with the same length into one
     *
     * To merge nested loops into one, use `merge` instead
     *
     * `parallelize`, `unroll` and `vectorize` properties will be reset on the
     * fused loop
     *
     * Suppose the original loops are labeled "L1" and "L2", the fused loop can
     * be selected by "$fuse{L1, L2}"
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
     * If moving a statement out of some loops, identical loops will be added
     * around the moved statement, which is equivalent to fission these loops
     *
     * @param stmt : ID of the statement to be moved
     * @param side : Whether `stmt` will be BEFORE or AFTER `dst
     * @param dst : Insert `stmt` to be directly after this statement
     * @throw InvalidSchedule if there is no feasible path to move
     * @return : (The new ID of the moved statement, The out-most newly
     * introduced statments including the added loops)
     */
    std::pair<ID, ID> moveTo(const ID &stmt, MoveToSide side, const ID &dst);

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
     * This schedule follows a fork-join model: multiple workers (abstract
     * threads) are created (but physically the threads may be cached in a
     * thread pool) when the loop begins, do their jobs in parallel, and join
     * when the loop ends
     *
     * OpenMP threads follow a typical fork-join model. CUDA threads run in a
     * bulk-synchronous parallel (BSP) model, which can also be mimiked by the
     * fork-join model: All threads start when the kernel get launched, but they
     * only begin to do their jobs when the parallel loop begins. Nevertheless,
     * the fork-join model needs the following extension to fully mimic a BSP
     * model:
     *
     * Taking CUDA as an example, we allow binding a loop to `threadIdx.x`
     * inside another loop bound to `threadIdx.x`, which is illegal in a classic
     * fork-join model. For example, we may implement a matmul with
     * collaborative fetch as below:
     *
     * ```
     * for i : threadIdx.x  # Li
     *   for j : threadIdx.y  # Lj
     *     local_sum = 0  # In gpu/local memory, unique to (i, j)
     *     for k0  # Lk0
     *       for k : threadIdx.y  # Lk1_a
     *         A_cache[k] = A[i, k]  # In gpu/shared, shared by different j
     *       for k : threadIdx.x  # Lk1_b
     *         B_cache[k] = B[k, j]  # In gpu/shared, shared by different i
     *       for k  # Lk1_c
     *         sum += A_cache[k] * B_cache[k]
     *     C[i, j] = local_sum
     * ```
     *
     * A seemingly plausible solution to avoid this extension is to reorder
     * `Lk0` to outer-most, and then move `Lk1_a` and `Lk1_b` out of `Li` or
     * `Lj`. This resolves the nested `threadIdx.x` and `threadIdx.y` binding
     * problem by running `Li+Lk1_a`, `Lj+Lk1_b` and `Li+Lj` interleavingly,
     * instead of running `Lk1_a` and `Lk1_b` inside `Li+Lj`. However, this
     * approach is illegal, because the local variable `local_sum` can no longer
     * be kept inside the body of `Li` and `Lj`: It has to be reused across
     * multiple runs of `Li` and `Lj`
     *
     * Please also note that we can bind one `threadIdx.x` to two loops only
     * when the body statement is loop-invariant to one of them. For example,
     * the following binding is still illegal, even in our extended fork-join
     * model, because it violates its serial semantics:
     *
     * ```
     * for i : threadIdx.x
     *   for j : threadIdx.x
     *     A[i, j] ++
     * ```
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
     * Use Pluto+ algorithm to permute and fuse two loops, with as most
     * parallelizable loops as possible at outermost levels.
     * The two loops are required to be consequent; all directly nested levels
     * are detected and subject to permutation. Remaining levels that cannot be
     * fused are left inside the fused loops as two statements
     *
     * @param loop0 : The first loop to fuse
     * @param loop1 : The second loop to fuse
     * @return std::pair<ID, int> : The ID of fused loop and level of
     * parallelizable loops
     */
    std::pair<ID, int> plutoFuse(const ID &loop0, const ID &loop1);
    
    /**
     * Use Pluto+ algorithm to permute a single loop, with as most
     * parallelizable loops as possible at outermost levels.
     *
     * @param loop : The loop to permute
     * @return std::pair<ID, int> : The ID of permuted loop and level of
     * parallelizable loops
     */
    std::pair<ID, int> plutoPermute(const ID &loop);

    /**
     * (Experimental) Automatic scheduling using some heuristics
     *
     * @param target : Target architecture
     * @param trace : Random decision tarce
     */
    void autoSchedule(const Target &target,
                      const Ref<RandTrace> &trace = nullptr);

    /**
     * (Experimental) Automatically use external libs using some heuristics
     *
     * @param target : Target architecture
     */
    void autoUseLib(const Target &target);

    /**
     * (Experimental) Automaticaly reorder loops in a loop nest
     *
     * @param target : Target architecture
     */
    void autoReorder(const Target &target);

    /**
     * (Experimental) Automatically fuse consecutive loops or vice versa using
     * some heuristics
     *
     * @param target : Target architecture
     * @param trace : Random decision tarce
     */
    void autoFissionFuse(const Target &target,
                         const Ref<RandTrace> &trace = nullptr);

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

    std::vector<AutoScheduleTuneTrial> tuneAutoSchedule(
        int nBatch, int batchSize, const Ref<Device> &device,
        const std::vector<Ref<Array>> &args,
        const std::unordered_map<std::string, Ref<Array>> &kws = {},
        const std::regex &toLearn = std::regex{".*"});

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
