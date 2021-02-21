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

bool StmtNode::hasNamedId() const { return id_.empty() || id_[0] != '#'; }

VarDefNode::VarDefNode(const VarDefNode &other)
    : name_(other.name_), buffer_(other.buffer_.clone()), body_(other.body_),
      infoAccLower_(other.infoAccLower_), infoAccLen_(other.infoAccLen_) {}

VarDefNode &VarDefNode::operator=(const VarDefNode &other) {
    name_ = other.name_;
    buffer_ = other.buffer_.clone();
    body_ = other.body_;
    infoAccLower_ = other.infoAccLower_;
    infoAccLen_ = other.infoAccLen_;
    return *this;
}

} // namespace ir

