#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <mutator.h>

namespace ir {

class ShrinkFor : public Mutator {
  protected:
    Stmt visit(const For &op) override;
};

Stmt shrinkFor(const Stmt &op);

} // namespace ir

#endif // SHRINK_FOR_H
