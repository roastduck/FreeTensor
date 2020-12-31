#include <stmt.h>

namespace ir {

uint64_t StmtNode::idCnt_ = 0;

void StmtNode::setId(const std::string &id) {
    if (id.empty()) {
        id_ = "#" + std::to_string(idCnt_++);
    } else {
        id_ = id;
    }
}

const std::string &StmtNode::id() const {
    ASSERT(!id_.empty());
    return id_;
}

VarDefNode::VarDefNode(const VarDefNode &other)
    : name_(other.name_), buffer_(other.buffer_.clone()) {}

VarDefNode &VarDefNode::operator=(const VarDefNode &other) {
    name_ = other.name_;
    buffer_ = other.buffer_.clone();
    return *this;
}

} // namespace ir

