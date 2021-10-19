#include <analyze/count_contig_access_loops.h>

namespace ir {

void CountContigAccessLoops::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    ASSERT(!var2for_.count(op->iter_));
    var2for_[op->iter_] = op;
    auto oldRepeat = repeat_;
    if (op->len_->nodeType() == ASTNodeType::IntConst) {
        repeat_ *= op->len_.as<IntConstNode>()->val_;
    } else {
        repeat_ *= 32; // guess at least 32 for unknown
    }
    depth_++;
    (*this)(op->body_);
    if (counts_.count(op->id())) {
        counts_[op->id()].second = -depth_;
    }
    depth_--;
    repeat_ = oldRepeat;
    var2for_.erase(op->iter_);
}

void CountContigAccessLoops::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    Visitor::visit(op);
    buffers_.erase(op->name_);
}

} // namespace ir

