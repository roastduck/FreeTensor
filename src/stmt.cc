#include <stmt.h>

namespace ir {

VarDefNode::VarDefNode(const VarDefNode &other)
    : name_(other.name_), buffer_(other.buffer_.clone()), body_(other.body_) {}

VarDefNode &VarDefNode::operator=(const VarDefNode &other) {
    name_ = other.name_;
    buffer_ = other.buffer_.clone();
    body_ = other.body_;
    return *this;
}

} // namespace ir
