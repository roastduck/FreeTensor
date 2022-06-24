#ifndef FREE_TENSOR_REORDER_H
#define FREE_TENSOR_REORDER_H

#include <string>
#include <vector>

#include <mutator.h>

namespace freetensor {

/**
 * Reorder two directly nested loops
 */
class Reorder : public Mutator {
    For oldOuter_, oldInner_;
    bool insideOuter_ = false, insideInner_ = false;
    bool visitedInner_ = false;

  public:
    Reorder(const For oldOuter, const For &oldInner)
        : oldOuter_(oldOuter), oldInner_(oldInner) {}

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

Stmt reorder(const Stmt &ast, const std::vector<ID> &order);

} // namespace freetensor

#endif // FREE_TENSOR_REORDER_H
