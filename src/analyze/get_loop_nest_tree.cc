#include <analyze/get_loop_nest_tree.h>

namespace freetensor {

void GetLoopNestTree::visit(const For &op) {
    Ref<LoopNest> loopNest = Ref<LoopNest>::make();
    loopNest->loop_ = op;
    parent_->subLoops_.emplace_back(loopNest);

    auto parent = parent_;
    parent_ = loopNest;
    Visitor::visit(op);
    parent_ = parent;
}

void GetLoopNestTree::visit(const Store &op) {
    Visitor::visit(op);
    parent_->leafStmts_.emplace_back(op);
}

void GetLoopNestTree::visit(const ReduceTo &op) {
    Visitor::visit(op);
    parent_->leafStmts_.emplace_back(op);
}

void GetLoopNestTree::visit(const Eval &op) {
    Visitor::visit(op);
    parent_->leafStmts_.emplace_back(op);
}

} // namespace freetensor
