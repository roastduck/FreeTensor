#ifndef FUSE_H
#define FUSE_H

#include <mutator.h>

namespace ir {

/**
 * Fuse two directly nested loops
 */
class FuseFor : public Mutator {
    For oldOuter_, oldInner_;
    Expr outerLen_, innerLen_;

    std::string newIter_, newId_;

    bool insideOuter_ = false, insideInner_ = false;
    bool visitedInner_ = false;

  public:
    FuseFor(const For oldOuter, const For &oldInner)
        : oldOuter_(oldOuter), oldInner_(oldInner),
          outerLen_(makeSub(oldOuter_->end_, oldOuter_->begin_)),
          innerLen_(makeSub(oldInner_->end_, oldInner_->begin_)),
          newIter_("f." + oldOuter_->iter_ + "." + oldInner_->iter_),
          newId_("fused." + oldOuter_->id_ + "." + oldInner_->id_) {}

    const std::string &newIter() const { return newIter_; }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Expr visit(const Var &op) override;
};

} // namespace ir

#endif // FUSE_H
