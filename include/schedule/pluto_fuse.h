#ifndef FREE_TENSOR_PLUTO_FUSE_H
#define FREE_TENSOR_PLUTO_FUSE_H

#include <stmt.h>

namespace freetensor {

std::pair<Stmt, ID> plutoFuse(const Stmt &ast, const ID &loop0, const ID &loop1,
                              int nestLevel = -1);

} // namespace freetensor

#endif
