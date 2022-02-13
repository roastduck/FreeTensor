#ifndef SWAP_H
#define SWAP_H

#include <string>
#include <vector>

#include <mutator.h>

namespace ir {

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

} // namespace ir

#endif // SWAP_H
