#include <analyze/count_contig_access_loops.h>

namespace ir {

void CountContigAccessLoops::visit(const For &op) {
    ASSERT(!var2for_.count(op->iter_));
    var2for_[op->iter_] = op;
    Visitor::visit(op);
    var2for_.erase(op->iter_);
}

} // namespace ir

