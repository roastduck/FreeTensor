#ifndef FREE_TENSOR_CHECK_NOT_IN_LIB_H
#define FREE_TENSOR_CHECK_NOT_IN_LIB_H

#include <analyze/find_stmt.h>
#include <except.h>

namespace freetensor {

inline void checkNotInLib(const Stmt &ast, const ID &stmt) {
    if (!findAllStmt(ast, "<MatMul>->>" + toString(stmt)).empty()) {
        throw InvalidSchedule("Scheduling " + toString(stmt) +
                              " inside a sub-program bound to an external "
                              "library call is meaningless");
    }
}

} // namespace freetensor

#endif // FREE_TENSOR_CHECK_NOT_IN_LIB_H
