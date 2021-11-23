#ifndef UNDO_MAKE_REDUCTION_H
#define UNDO_MAKE_REDUCTION_H

#include <mutator.h>

namespace ir {

/**
 * Transform things like a += b into a = a + b
 */
class UndoMakeReduction : public Mutator {
  protected:
    Stmt visit(const ReduceTo &op) override;
};

inline Stmt undoMakeReduction(const Stmt &op) {
    return UndoMakeReduction()(op);
}

} // namespace ir

#endif // UNDO_MAKE_REDUCTION_H
