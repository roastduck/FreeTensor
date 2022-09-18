#ifndef FREE_TENSOR_PLUTO_FUSE_H
#define FREE_TENSOR_PLUTO_FUSE_H

#include <stmt.h>

namespace freetensor {

std::pair<Stmt, std::pair<ID, int>> plutoFuse(const Stmt &ast, const ID &loop0, const ID &loop1);

} // namespace freetensor

#endif
