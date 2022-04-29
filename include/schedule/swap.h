#ifndef FREE_TENSOR_SWAP_H
#define FREE_TENSOR_SWAP_H

#include <string>
#include <vector>

#include <mutator.h>

namespace freetensor {

class Swap : public Mutator {
    std::vector<ID> order_;
    StmtSeq scope_;

  public:
    Swap(const std::vector<ID> &order) : order_(order) {}

    const StmtSeq &scope() const { return scope_; }

  protected:
    Stmt visit(const StmtSeq &op) override;
};

Stmt swap(const Stmt &ast, const std::vector<ID> &order);

} // namespace freetensor

#endif // FREE_TENSOR_SWAP_H
