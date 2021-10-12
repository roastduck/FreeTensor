#ifndef GET_LOOP_NEST_TREE_H
#define GET_LOOP_NEST_TREE_H

#include <visitor.h>

namespace ir {

struct LoopNest {
    For loop_;
    std::vector<Ref<LoopNest>> subLoops_;
};

class GetLoopNestTree : public Visitor {
    Ref<LoopNest> root_, parent_;

  public:
    GetLoopNestTree() : root_(Ref<LoopNest>::make()), parent_(root_) {}

    Ref<LoopNest> result() const { return root_; }

  protected:
    void visit(const For &op) override;
};

inline Ref<LoopNest> getLoopNestTree(const Stmt &op) {
    GetLoopNestTree visitor;
    visitor(op);
    return visitor.result();
}

} // namespace ir

#endif // GET_LOOP_NEST_TREE_H
