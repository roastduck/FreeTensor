#ifndef FREE_TENSOR_UNDO_MAKE_REDUCTION_H
#define FREE_TENSOR_UNDO_MAKE_REDUCTION_H

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

/**
 * Transform things like a += b into a = a + b
 *
 * Transform a statement
 */
Stmt undoMakeReduction(const ReduceTo &op, DataType dtype);

class UndoMakeReduction : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

  protected:
    using BaseClass::visit;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Transform things like a += b into a = a + b
 *
 * Transform a whole AST
 */
inline Stmt undoMakeReduction(const Stmt &op) {
    return UndoMakeReduction()(op);
}

} // namespace freetensor

#endif // FREE_TENSOR_UNDO_MAKE_REDUCTION_H
