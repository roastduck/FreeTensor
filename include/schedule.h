#ifndef SCHEDULE_H
#define SCHEDULE_H

#include <functional>
#include <unordered_map>

#include <cursor.h>
#include <driver/target.h>
#include <func.h>
#include <schedule/var_split.h>
#include <stmt.h>

namespace ir {

enum MoveToSide : int { Before, After };

class Schedule {
    Func func_;
    Stmt ast_;

    std::vector<std::string> logs_;

  public:
    Schedule() = default;
    Schedule(const Stmt &ast);
    Schedule(const Func &func) : Schedule(func->body_) { func_ = func; }

    Schedule clone() const { return Schedule(deepCopy(func_)); }

    /**
     * @return : The function being transformed
     */
    Func func() const {
        ASSERT(func_.isValid());
        return makeFunc(func_->name_, func_->params_, ast_, func_->src_);
    }

    /**
     * @return : The statements being transformed, without a function signature
     */
    Stmt ast() const { return ast_; }

    /**
     * @return : Logs of all schedules applied
     */
    std::vector<std::string> logs() const { return logs_; }

    /**
     * Find all nodes in the current AST satisfying a given condition
     * @param filter : A callback. Return true for acceptance
     */
    std::vector<Cursor>
    findAll(const std::function<bool(const Cursor &)> &filter) const;

    /**
     * Find the only one nodes in the current AST satisfying a given condition
     * @param filter : A callback. Return true for acceptance
     * @throw Error : if there is more than one, or there is no node found
     */
    Cursor find(const std::function<bool(const Cursor &)> &filter) const;

    /**
     * Find a (maybe non-existing) node in the current AST by ID
     * @param id: ID
     */
    std::vector<Cursor> findAll(const std::string &id) const {
        return findAll([&id](const Cursor &c) { return c.id() == id; });
    }

    /**
     * Find a node in the current AST by ID
     * @param id: ID
     */
    Cursor find(const std::string &id) const {
        return find([&id](const Cursor &c) { return c.id() == id; });
    }

    /**
     * Split a loop into two nested loops
     *
     * To fission a loop into two consecutive loops, use `fission` instead
     *
     * @param id : ID of the loop to be split
     * @param factor : Length of the inner loop. Set to -1 if using `nparts`
     * @param nparts : Length of the outer loop. Set to -1 if using `factor`
     * @throw InvalidSchedule if the loop is not found
     * @return : (outer loop ID, inner loop ID)
     */
    std::pair<std::string, std::string> split(const std::string &id,
                                              int factor = -1, int nparts = -1);

    /**
     * Reorder directly nested loops
     *
     * To swap consecutive loops, use `swap` instead
     *
     * @param order : Vector of loop IDs. The requested order of the loops
     * @throw InvalidSchedule if the input is invalid or there are breaking
     * dependencies
     */
    void reorder(const std::vector<std::string> &order);

    /**
     * Merge two directly nested loops into one
     *
     * To fuse consecutive loops, use `fuse` instead
     *
     * `parallelize`, `unroll` and `vectorize` properties will be reset on the
     * fused loop
     *
     * @param loop1, loop2 : ID of the loops to be merged, can be in any order
     * @throw InvalidSchedule if the loops are not directly nested
     * @return : ID of the merged loop
     */
    std::string merge(const std::string &loop1, const std::string &loop2);

    typedef std::unordered_map<std::string, std::string> IDMap;
    /**
     * Fission a loop into two loops each containing part of the statements, one
     * followed by another
     *
     * To split loop into two nested loops, use `split` instead
     *
     * @param loop : ID of the loop to be fissioned
     * @param after : ID of the last statement of the first loop
     * @param suffix0 : ID suffix of the statements in the first loop, default
     * to ".a", can be "" for convenience, but cannot be the same with suffix1
     * @param suffix1 : ID suffix of the statements in the second loop, default
     * to ".b", can be "" for convenience, but cannot be the same with suffix0
     * @throw InvalidSchedule if any dependency cannot be resolved
     * @return : ({old ID -> new ID in 1st loop}, {old ID -> new ID in 2nd
     * loop})
     */
    std::pair<IDMap, IDMap> fission(const std::string &loop,
                                    const std::string &after,
                                    const std::string &suffix0 = ".a",
                                    const std::string &suffix1 = ".b");

    /**
     * Fuse two directly following loops with the same length into one
     *
     * To merge nested loops into one, use `merge` instead
     *
     * @param loop0 : ID of the leading loop
     * @param loop1 : ID of the following loop
     * @throw InvalidSchedule if the two loops are not directly following, the
     * two loops are not with the same length, or there is any dependency cannot
     * be resolved
     * @return : ID of the result loop
     */
    std::string fuse(const std::string &loop0, const std::string &loop1);

    /**
     * Swap statements in the same block
     *
     * To reorder nested loops, use `reorder` instead
     *
     * @param order : list of IDs of the statements
     * @throw InvalidSchedule if the statements are not found or the
     * dependencies cannot be solved
     */
    void swap(const std::vector<std::string> &order);

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
    void blend(const std::string &loop);

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
    std::tuple<std::string, std::string, std::string, std::string>
    cache(const std::string &stmt, const std::string &var, MemType mtype);

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
    std::tuple<std::string, std::string, std::string, std::string>
    cacheReduction(const std::string &stmt, const std::string &var,
                   MemType mtype);

    /**
     * Change where a variable is stored
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param mtype : Where the variable should be stored
     * @throw InvalidSchedule if the variable is not found
     */
    void setMemType(const std::string &def, MemType mtype);

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
    void varSplit(const std::string &def, int dim, VarSplitMode mode,
                  int factor = -1, int nparts = -1);

    /**
     * Reorder the dimensions of a variable
     *
     * @param def : ID of the VarDef statement of the specific variable
     * @param order : new order of the dimensions
     * @throw InvalidSchedule if the variable or the order is illegal
     */
    void varReorder(const std::string &def, const std::vector<int> &order);

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
    std::string moveTo(const std::string &stmt, MoveToSide side,
                       const std::string &dst);

    /**
     * Remove a variable. When the variable is used, recompute its value
     *
     * @param def : ID of the VarDef statement of the specific variable. It can
     * not be an I/O varible
     * @throw InvalidSchedule if the variable cannot be completely removed
     */
    void inlining(const std::string &def);

    /**
     * Mark a loop with a parallel implementation
     *
     * @param loop : ID of the loop
     * @param parallel : Parallel implementation. Supported values are "openmp",
     * "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y",
     * "threadIdx.z"
     */
    void parallelize(const std::string &loop, const std::string &parallel);

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
    void unroll(const std::string &loop, bool immediate = false);

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
    void vectorize(const std::string &loop);

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
     * Each loop will be seperated into 2 parts: the body and the tail. After
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
     * If there is two VarDef nodes in two branches, it may result in doubled
     * memory use, since different thread may go to different branch. Therefore,
     * this pass will not duplicate VarDef nodes. (TODO: This restriction may be
     * limited to non-local buffers)
     *
     * Ideally, all programs can benefit from this schedule. However, this
     * schedule may greatly increase the program size and make the compiling
     * time way too long. Therefore, this transformation is implemented as a
     * schedule, which can be applied optionally. (TODO: Optionally apply this
     * schedule to part of the program)
     */
    void seperateTail();

    /**
     * Transform nested loops to be a external call to a matrix multiplication
     *
     * @param loop: ID of the loop
     * @throw InvalidSchedule if the loop cannot be transformed to be a matrix
     * multiplication
     */
    void asMatMul(const std::string &loop);

    /**
     * Automatic scheduling using some heuristics
     */
    void autoSchedule(const Target &target);

    /**
     * Automatically parallelize some loops using some heuristics
     *
     * @param target : Target architecture
     */
    void autoParallelize(const Target &target);

    /**
     * Automatically set memory types using some heuristics
     */
    void autoSetMemType(const Target &target);
};

} // namespace ir

#endif // SCHEDULE_H
