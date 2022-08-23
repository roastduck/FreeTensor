#ifndef FREE_TENSOR_GET_LOOP_NEST_TREE_H
#define FREE_TENSOR_GET_LOOP_NEST_TREE_H

#include <visitor.h>

namespace freetensor {

struct LoopNest {
    For loop_;
    std::vector<Ref<LoopNest>> subLoops_;
    std::vector<Stmt> leafStmts_; // Store, ReduceTo and Eval nodes nested in
                                  // this loop but not in its children loops
};

class GetLoopNestTree : public Visitor {
    Ref<LoopNest> root_, parent_;

  public:
    GetLoopNestTree() : root_(Ref<LoopNest>::make()), parent_(root_) {}

    Ref<LoopNest> result() const { return root_; }

  protected:
    void visit(const For &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Eval &op) override;
    void visit(const MatMul &op) override {} // do nothing
};

inline Ref<LoopNest> getLoopNestTree(const Stmt &op) {
    GetLoopNestTree visitor;
    visitor(op);
    return visitor.result();
}

} // namespace freetensor

#endif // FREE_TENSOR_GET_LOOP_NEST_TREE_H
