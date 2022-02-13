#include <analyze/get_loop_nest_tree.h>

namespace ir {

void GetLoopNestTree::visit(const For &op) {
    Ref<LoopNest> loopNest = Ref<LoopNest>::make();
    loopNest->loop_ = op;
    parent_->subLoops_.emplace_back(loopNest);

    auto parent = parent_;
    parent_ = loopNest;
    Visitor::visit(op);
    parent_ = parent;
}

} // namespace ir
