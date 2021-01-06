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
     * @param order : Vector of loop IDs. The requested order of the loops
     * @throw InvalidSchedule if the input is invalid or there are breaking
     * dependencies
     */
    void reorder(const std::vector<std::string> &order);

    /**
     * Merge two directly nested loops into one
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
     * @param loop : ID of the loop to be fissioned
     * @param after : ID of the last statment of the first loop
     * @throw InvalidSchedule if any dependency cannot be resolved
     * @return : ({old ID -> new ID in 1st loop}, {old ID -> new ID in 2nd
     * loop})
     */
    std::pair<IDMap, IDMap> fission(const std::string &loop,
                                    const std::string &after);

    /**
     * Fuse two directly following loops with the same length into one
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
     * Cache the reads of a variable into a new local variable
     *
     * @param stmt : ID of the statment or block (e.g. an If) to be modified.
     * Note that it is not supported to define a local variable inside stmt and
     * use it to read the variable to be cached.
     * @param var : name of the variable to be cached
     * @throw InvalidSchedule if the ID or name is not found
     * @return : (ID of the statment that fills into the cache, name of the
     * cache variable)
     */
    std::pair<std::string, std::string> cacheRead(const std::string &stmt,
                                                  const std::string &var);

    /**
     * Cache the writes of a variable into a new local variable
     *
     * @param stmt : ID of the statment or block (e.g. an If) to be modified
     * Note that it is not supported to define a local variable inside stmt and
     * use it to write the variable to be cached.
     * @param var : name of the variable to be cached
     * @throw InvalidSchedule if the ID or name is not found
     * @return : (ID of the statment that flushes from the cache, name of the
     * cache variable)
     */
    std::pair<std::string, std::string> cacheWrite(const std::string &stmt,
                                                   const std::string &var);
};

} // namespace ir

#endif // SCHEDULE_H
