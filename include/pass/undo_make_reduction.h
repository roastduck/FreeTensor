#ifndef FREE_TENSOR_UNDO_MAKE_REDUCTION_H
#define FREE_TENSOR_UNDO_MAKE_REDUCTION_H

#include <mutator.h>

namespace freetensor {

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

} // namespace freetensor

#endif // FREE_TENSOR_UNDO_MAKE_REDUCTION_H
