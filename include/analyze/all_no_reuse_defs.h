#ifndef ALL_NO_REUSE_DEFS_H
#define ALL_NO_REUSE_DEFS_H

#include <unordered_set>

#include <stmt.h>

namespace ir {

std::vector<ID> allNoReuseDefs(const Stmt &op,
                               const std::unordered_set<AccessType> &atypes);

}

#endif // ALL_NO_REUSE_DEFS_H
