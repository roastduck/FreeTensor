#ifndef REORDER_H
#define REORDER_H

#include <string>
#include <vector>

#include <mutator.h>
#include <visitor.h>

namespace ir {

/**
 * Transform a = a + b into a += b
 *
 * This is to make the dependency analysis more accurate
 */
class MakeReduction : public Mutator {
  private:
    bool isSameElem(const Store &s, const Load &l);

  protected:
    Stmt visit(const Store &op) override;
};

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
