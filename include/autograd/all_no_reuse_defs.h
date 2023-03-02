#ifndef FREE_TENSOR_ALL_NO_REUSE_DEFS_H
#define FREE_TENSOR_ALL_NO_REUSE_DEFS_H

#include <unordered_set>

#include <stmt.h>

namespace freetensor {

std::vector<ID> allNoReuseDefs(const Stmt &op,
                               const std::unordered_set<AccessType> &atypes);

}

#endif // FREE_TENSOR_ALL_NO_REUSE_DEFS_H
