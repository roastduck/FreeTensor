#include <analyze/count_contig_access_loops.h>

namespace ir {

void CountContigAccessLoops::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    auto oldRepeat = repeat_;
    if (op->len_->nodeType() == ASTNodeType::IntConst) {
        repeat_ *= op->len_.as<IntConstNode>()->val_;
    } else {
        repeat_ *= 32; // guess at least 32 for unknown
    }
    depth_++;
    pushFor(op);
    (*this)(op->body_);
    popFor(op);
    if (counts_.count(op->id())) {
        counts_[op->id()].second = -depth_;
    }
    depth_--;
    repeat_ = oldRepeat;
}

} // namespace ir
