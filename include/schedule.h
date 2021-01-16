#ifndef SCHEDULE_H
#define SCHEDULE_H

#include <unordered_map>

#include <stmt.h>

namespace ir {

class Schedule {
    Stmt ast_;

  public:
    Schedule(const Stmt &ast) : ast_(ast) {}

    const Stmt &ast() const { return ast_; }

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
                                    const std::string &suffx0 = ".a",
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
     * Cache the reads of a variable into a new local variable
     *
     * @param stmt : ID of the statement or block (e.g. an If) to be modified.
     * Note that it is not supported to define a local variable inside stmt and
     * use it to read the variable to be cached.
     * @param var : name of the variable to be cached
     * @throw InvalidSchedule if the ID or name is not found
     * @return : (ID of the statement that fills into the cache, name of the
     * cache variable)
     */
    std::pair<std::string, std::string> cacheRead(const std::string &stmt,
                                                  const std::string &var);

    /**
     * Cache the writes of a variable into a new local variable
     *
     * @param stmt : ID of the statement or block (e.g. an If) to be modified
     * Note that it is not supported to define a local variable inside stmt and
     * use it to write the variable to be cached.
     * @param var : name of the variable to be cached
     * @throw InvalidSchedule if the ID or name is not found
     * @return : (ID of the statement that flushes from the cache, name of the
     * cache variable)
     */
    std::pair<std::string, std::string> cacheWrite(const std::string &stmt,
                                                   const std::string &var);

    /**
     * Move a statement to a new position
     *
     * This is a composite schedule command, which is implemented with other
     * commands
     *
     * @param stmt : ID of the statement to be moved
     * @param dst : Insert `stmt` to be directly after this statement
     * @param toBegin : Move to the begin point of dst. This specifies the
     * behavior when dst is a scope, e.g. a loop
     * @param toEnd : Move to the end point of dst This specifies the behavior
     * when dst is a scope, e.g. a loop
     * @throw InvalidSchedule if there is no feasible path to move
     * @return : The new ID of stmt
     */
    std::string moveTo(const std::string &stmt, const std::string &dst,
                       bool toBegin = false, bool toEnd = false);

    /**
     * Mark a loop with a parallel implementation
     *
     * @param loop : ID of the loop
     * @param parallel : Parallel implementation. Supported values are "openmp",
     * "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y",
     * "threadIdx.z"
     */
    void parallelize(const std::string &loop, const std::string &parallel);
};

} // namespace ir

#endif // SCHEDULE_H
