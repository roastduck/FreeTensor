#ifndef REORDER_H
#define REORDER_H

#include <string>
#include <vector>

#include <mutator.h>

namespace ir {

/**
 * Swap two directly nested loops
 */
class SwapFor : public Mutator {
    For oldOuter_, oldInner_;
    bool insideOuter_ = false, insideInner_ = false;
    bool visitedInner_ = false;

  public:
    SwapFor(const For oldOuter, const For &oldInner)
        : oldOuter_(oldOuter), oldInner_(oldInner) {}

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

} // namespace ir

#endif // REORDER_H
