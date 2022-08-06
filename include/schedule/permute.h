#ifndef FREE_TENSOR_PERMUTE_H
#define FREE_TENSOR_PERMUTE_H

#include <string>
#include <vector>

#include <mutator.h>

namespace freetensor {

std::pair<Stmt, std::vector<ID>> permute(
    const Stmt &ast, const std::vector<ID> &loopsId,
    const std::function<std::vector<Expr>(std::vector<Expr>)> &transformFunc);

} // namespace freetensor

#endif // FREE_TENSOR_PERMUTE_H
