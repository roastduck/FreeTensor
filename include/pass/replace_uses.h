#ifndef FREE_TENSOR_REPLACE_USES_H
#define FREE_TENSOR_REPLACE_USES_H

#include <unordered_map>

#include <mutator.h>

namespace freetensor {

/**
 * Replace load to some other expression, or replace reductions like `a += b` to
 * `a = (some other expression) + b`
 */
class ReplaceUses : public Mutator {
    const std::unordered_map<AST, Expr> &replace_;

  public:
    ReplaceUses(const std::unordered_map<AST, Expr> &replace)
        : replace_(replace) {}

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_REPLACE_USES_H
