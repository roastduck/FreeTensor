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
    std::string id0_, id1_, fused_, iter0_, iter1_, beforeId_, afterId_;
    Expr begin0_, begin1_;
    bool inLoop0_ = false, inLoop1_ = false;

  public:
    FuseFor(const std::string &id0, const std::string &id1)
        : id0_(id0), id1_(id1), fused_("fused." + id0 + "." + id1) {}

    const std::string &fused() const { return fused_; }
    const std::string &beforeId() const { return beforeId_; }
    const std::string &afterId() const { return afterId_; }

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

class CheckAccessible : public Visitor {
    std::string id0_, id1_;
    LoopInVarDefs loop0InVarDefs_, loop1InVarDefs_;

  public:
    CheckAccessible(const std::string &id0, const std::string &id1)
        : id0_(id0), id1_(id1) {}

    const LoopInVarDefs &loop0() const { return loop0InVarDefs_; }
    const LoopInVarDefs &loop1() const { return loop1InVarDefs_; }

  protected:
    void visit(const StmtSeq &op) override;
};

std::pair<Stmt, std::string> fuse(const Stmt &ast, const std::string &loop0,
                                  const std::string &loop1);

} // namespace ir

#endif // FUSE_H
