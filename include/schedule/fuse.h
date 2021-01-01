#ifndef FUSE_H
#define FUSE_H

#include <mutator.h>

namespace ir {

class FuseFor : public Mutator {
    std::string id0_, id1_, fused_, iter0_, iter1_;
    Expr begin0_, begin1_;

  public:
    FuseFor(const std::string &id0, const std::string &id1)
        : id0_(id0), id1_(id1), fused_("fused." + id0 + "." + id1) {}

    const std::string &fused() const { return fused_; }

  protected:
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

} // namespace ir

#endif // FUSE_H
