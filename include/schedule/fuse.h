#ifndef FREE_TENSOR_FUSE_H
#define FREE_TENSOR_FUSE_H

#include <mutator.h>
#include <visitor.h>

namespace freetensor {

struct LoopInScopes {
    For loop_;
    std::vector<Stmt> scopes_; // inner to outer
};

enum class FindLoopInScopesDirection : int { Front, Back };

class FuseFor : public Mutator {
    Stmt root_;
    ID id0_, id1_, fused_;
    std::string iter0_, iter1_, newIter_;
    ID beforeId_, afterId_;
    Expr begin0_, begin1_, step0_, step1_;
    bool strict_, inLoop0_ = false, inLoop1_ = false;

  public:
    FuseFor(const Stmt &root, const ID &id0, const ID &id1,
            const std::string &newIter, bool strict)
        : root_(root), id0_(id0), id1_(id1), newIter_(newIter),
          strict_(strict) {}

    const ID &fused() const { return fused_; }
    const ID &beforeId() const { return beforeId_; }
    const ID &afterId() const { return afterId_; }

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

class CheckFuseAccessible : public Visitor {
    ID id0_, id1_;
    LoopInScopes loop0InScopes_, loop1InScopes_;

  public:
    CheckFuseAccessible(const ID &id0, const ID &id1) : id0_(id0), id1_(id1) {}

    const LoopInScopes &loop0() const { return loop0InScopes_; }
    const LoopInScopes &loop1() const { return loop1InScopes_; }

    void check(const Stmt &ast);

  protected:
    void visit(const StmtSeq &op) override;
};

std::pair<Stmt, ID> fuse(const Stmt &ast, const ID &loop0, const ID &loop1,
                         bool strict);

} // namespace freetensor

#endif // FREE_TENSOR_FUSE_H
