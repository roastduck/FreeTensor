#include <stmt.h>

namespace ir {

VarDefNode::VarDefNode(const VarDefNode &other)
    : name_(other.name_), buffer_(other.buffer_.clone()),
      sizeLim_(other.sizeLim_), body_(other.body_), pinned_(other.pinned_) {}

VarDefNode &VarDefNode::operator=(const VarDefNode &other) {
    name_ = other.name_;
    buffer_ = other.buffer_.clone();
    sizeLim_ = other.sizeLim_;
    body_ = other.body_;
    pinned_ = other.pinned_;
    return *this;
}

} // namespace ir
