#include <algorithm>

#include <itertools.hpp>

#include <schedule/check_loop_order.h>

namespace ir {

void CheckLoopOrder::visit(const For &op) {
    if (done_) {
        return;
    }
    if (std::find(dstOrder_.begin(), dstOrder_.end(), op->id()) !=
        dstOrder_.end()) {
        curOrder_.emplace_back(op);
        if (curOrder_.size() < dstOrder_.size()) {
            Visitor::visit(op);
        }
        done_ = true;
        // done_ is to avoid such a program:
        // for i {
        //	 for j {}
        //	 for k {}
        // }
    } else {
        if (!curOrder_.empty()) { // Already met the first loop
            throw InvalidSchedule("Unable to find all the loops. "
                                  "These loops should be directly nested");
        }
        Visitor::visit(op);
    }
}

const std::vector<For> &CheckLoopOrder::order() const {
    if (curOrder_.size() != dstOrder_.size()) {
        std::string msg = "Loops ";
        for (auto &&[i, item] : iter::enumerate(dstOrder_)) {
            msg += (i > 0 ? ", " : "") + toString(item);
        }
        msg += "should be directly nested";
        throw InvalidSchedule(msg);
    }
    return curOrder_;
}

} // namespace ir

