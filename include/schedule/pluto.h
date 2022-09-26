#ifndef FREE_TENSOR_PLUTO_H
#define FREE_TENSOR_PLUTO_H

#include <stmt.h>

namespace freetensor {

std::pair<Stmt, std::pair<ID, int>> plutoFuse(const Stmt &ast, const ID &loop0,
                                              const ID &loop1);
std::pair<Stmt, std::pair<ID, int>> plutoPermute(const Stmt &ast,
                                                 const ID &loop);

} // namespace freetensor

#endif
