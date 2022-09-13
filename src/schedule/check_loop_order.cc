#include <algorithm>

#include <container_utils.h>
#include <schedule/check_loop_order.h>

namespace freetensor {

void CheckLoopOrder::visitStmt(const Stmt &stmt) {
    if (done_) {
        return;
    }
    Visitor::visitStmt(stmt);
}

void CheckLoopOrder::visit(const For &op) {
    if (std::find(dstOrder_.begin(), dstOrder_.end(), op->id()) !=
        dstOrder_.end()) {
        curOrder_.emplace_back(op);
        if (curOrder_.size() < dstOrder_.size()) {
            Visitor::visit(op);
        }
        if (!done_) {
            done_ = true;
            outerLoops_ = outerLoopStack_;
            stmtSeqInBetween_ = stmtSeqStack_;
        }
        // done_ is to avoid such a program:
        // for i {
        //	 for j {}
        //	 for k {}
        // }
    } else if (curOrder_.empty()) {
        // not yet started
        outerLoopStack_.emplace_back(op);
        Visitor::visit(op);
        outerLoopStack_.pop_back();
    }
}

void CheckLoopOrder::visit(const StmtSeq &op) {
    if (curOrder_.empty()) {
        Visitor::visit(op);
    } else {
        stmtSeqStack_.emplace_back(op);
        Visitor::visit(op);
        stmtSeqStack_.pop_back();
    }
}

const std::vector<For> &CheckLoopOrder::order() const {
    if (curOrder_.size() != dstOrder_.size()) {
        std::string msg = "Loops ";
        for (auto &&[i, item] : views::enumerate(dstOrder_)) {
            msg += (i > 0 ? ", " : "") + toString(item);
        }
        msg += " should be directly nested";
        throw InvalidSchedule(msg);
    }
    return curOrder_;
}

} // namespace freetensor
