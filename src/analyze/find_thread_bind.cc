#include <analyze/find_thread_bind.h>

namespace ir {
void FindThreadBind::visit(const For &op) {
    if (!downward_ && !stackMarkBranch_.empty()) {
        stackMarkBranch_.back() = true;
    }
    stackId_.push_back(
        std::make_pair(op->id(), op->len_.as<IntConstNode>()->val_));
    stackMarkBranch_.push_back(false);
    downward_ = true;
    Visitor::visit(op);
    if (stackMarkBranch_.back()) {
        bufId_.clear();
    }
    bufId_.push_back(stackId_.back());
    stackId_.pop_back();
    stackMarkBranch_.pop_back();
    downward_ = false;
}

std::vector<std::pair<std::string, int>> FindThreadBind::result() {
    found_.clear();
    while (!bufId_.empty()) {
        found_.push_back(bufId_.back());
        bufId_.pop_back();
    }
    return found_;
}

} // namespace ir