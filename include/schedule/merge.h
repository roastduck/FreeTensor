#ifndef FREE_TENSOR_MERGE_H
#define FREE_TENSOR_MERGE_H

#include <mutator.h>

namespace freetensor {

/**
 * Merge two directly nested loops
 */
class MergeFor : public Mutator {
    Stmt root_;

    For oldOuter_, oldInner_;
    Expr outerLen_, innerLen_;

    std::string newIter_;
    ID newId_;

    bool insideOuter_ = false, insideInner_ = false;
    bool visitedInner_ = false;

  public:
    MergeFor(const Stmt &root, const For oldOuter, const For &oldInner)
        : root_(root), oldOuter_(oldOuter), oldInner_(oldInner),
          outerLen_(oldOuter_->len_), innerLen_(oldInner_->len_),
          newIter_("m." + oldOuter_->iter_ + "." + oldInner_->iter_) {}

    const std::string &newIter() const { return newIter_; }
    const ID &newId() const { return newId_; }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Expr visit(const Var &op) override;
    Stmt visit(const VarDef &op) override;
};

std::pair<Stmt, ID> merge(const Stmt &ast, const ID &loop1, const ID &loop2);

} // namespace freetensor

#endif // FREE_TENSOR_MERGE_H
