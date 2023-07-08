#ifndef FREE_TENSOR_REORDER_H
#define FREE_TENSOR_REORDER_H

#include <string>
#include <vector>

#include <mutator.h>

namespace freetensor {

enum class ReorderMode : int { PerfectOnly, MoveOutImperfect, MoveInImperfect };
inline std::ostream &operator<<(std::ostream &os, ReorderMode mode) {
    switch (mode) {
    case ReorderMode::PerfectOnly:
        return os << "perfect_only";
    case ReorderMode::MoveOutImperfect:
        return os << "move_out_imperfect";
    case ReorderMode::MoveInImperfect:
        return os << "move_in_imperfect";
    default:
        ASSERT(false);
    }
}

class RenameIter : public Mutator {
    std::string oldName_, newName_;

  public:
    RenameIter(const std::string &oldName) : oldName_(oldName) {}

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
};

/**
 * Reorder two directly nested loops
 */
class Reorder : public Mutator {
    For oldOuter_, oldInner_;
    ReorderMode mode_;
    bool insideOuter_ = false, insideInner_ = false;
    bool visitedInner_ = false;

  public:
    Reorder(const For oldOuter, const For &oldInner, ReorderMode mode)
        : oldOuter_(oldOuter), oldInner_(oldInner), mode_(mode) {}

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

Stmt reorder(const Stmt &ast, const std::vector<ID> &order, ReorderMode mode);

} // namespace freetensor

#endif // FREE_TENSOR_REORDER_H
