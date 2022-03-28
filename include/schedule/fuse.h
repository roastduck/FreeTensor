#ifndef FUSE_H
#define FUSE_H

#include <mutator.h>
#include <visitor.h>

namespace ir {

struct LoopInVarDefs {
    For loop_;
    std::vector<Stmt> surroundings_; // inner to outer
};

enum class FindLoopInVarDefsDirection : int { Front, Back };

class FuseFor : public Mutator {
    Stmt root_;
    ID id0_, id1_, fused_;
    std::string iter0_, iter1_;
    ID beforeId_, afterId_;
    Expr begin0_, begin1_, step0_, step1_;
    bool strict_, inLoop0_ = false, inLoop1_ = false;

  public:
    FuseFor(const Stmt &root, const ID &id0, const ID &id1, bool strict)
        : root_(root), id0_(id0), id1_(id1),
          fused_("fused." + id0.strId() + "." + id1.strId()), strict_(strict) {}

    const ID &fused() const { return fused_; }
    const ID &beforeId() const { return beforeId_; }
    const ID &afterId() const { return afterId_; }

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

class CheckAccessible : public Visitor {
    ID id0_, id1_;
    LoopInVarDefs loop0InVarDefs_, loop1InVarDefs_;

  public:
    CheckAccessible(const ID &id0, const ID &id1) : id0_(id0), id1_(id1) {}

    const LoopInVarDefs &loop0() const { return loop0InVarDefs_; }
    const LoopInVarDefs &loop1() const { return loop1InVarDefs_; }

  protected:
    void visit(const StmtSeq &op) override;
};

std::pair<Stmt, ID> fuse(const Stmt &ast, const ID &loop0, const ID &loop1,
                         bool strict);

} // namespace ir

#endif // FUSE_H
