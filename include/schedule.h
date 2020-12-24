#ifndef SCHEDULE_H
#define SCHEDULE_H

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
     * @return : (outer loop ID, inner loop ID)
     */
    std::pair<std::string, std::string> split(const std::string &id,
                                              int factor = -1, int nparts = -1);
};

} // namespace ir

#endif // SCHEDULE_H
