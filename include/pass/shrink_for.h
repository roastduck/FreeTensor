#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <mutator.h>

namespace ir {

class ShrinkFor : public Mutator {
    bool keepConst_;

  public:
    ShrinkFor(bool keepConst) : keepConst_(keepConst) {}

  protected:
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 *
 * @param keepConst : If true, do not transform loops to have variable begins
 * and ends.
 */
Stmt shrinkFor(const Stmt &op, bool keepConst = false);

} // namespace ir

#endif // SHRINK_FOR_H
