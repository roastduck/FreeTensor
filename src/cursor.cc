#include <algorithm>

#include <cursor.h>

namespace ir {

Stmt Cursor::getParentById(const std::string &id) const {
    for (auto it = stack_.top(); it.isValid(); it = it->prev_) {
        if (it->data_->id() == id) {
            return it->data_;
        }
    }
    return nullptr;
}

bool Cursor::isBefore(const Cursor &other) const {
    auto l = stack_, r = other.stack_;
    while (l.size() > r.size()) {
        l.pop();
    }
    while (r.size() > l.size()) {
        r.pop();
    }
    for (; !l.empty(); l.pop(), r.pop()) {
        if (l.top()->data_->id() == r.top()->data_->id()) {
            return false;
        }
        if (l.top()->prev_.isValid() && r.top()->prev_.isValid() &&
            l.top()->prev_->data_->id() == r.top()->prev_->data_->id()) {
            auto &&prev = l.top()->prev_->data_;
            ASSERT(prev->nodeType() == ASTNodeType::StmtSeq);
            auto seq = prev.as<StmtSeqNode>();
            auto il = std::find_if(
                seq->stmts_.begin(), seq->stmts_.end(),
                [&](const Stmt &s) { return s->id() == l.top()->data_->id(); });
            auto ir = std::find_if(
                seq->stmts_.begin(), seq->stmts_.end(),
                [&](const Stmt &s) { return s->id() == r.top()->data_->id(); });
            return il < ir;
        }
    }
    return false;
}

bool Cursor::isOuter(const Cursor &other) const {
    auto l = stack_, r = other.stack_;
    if (r.size() <= l.size()) {
        return false;
    }
    while (r.size() > l.size()) {
        r.pop();
    }
    return l.empty() || l.top()->data_->id() == r.top()->data_->id();
}

bool Cursor::hasPrev() const {
    if (!stack_.top()->prev_.isValid()) {
        return false;
    }
    if (stack_.top()->prev_->data_->nodeType() != ASTNodeType::StmtSeq) {
        return false;
    }
    auto seq = stack_.top()->prev_->data_.as<StmtSeqNode>();
    auto it = std::find_if(seq->stmts_.begin(), seq->stmts_.end(),
                           [&](const Stmt &s) { return s->id() == id(); });
    return it > seq->stmts_.begin();
}

Cursor Cursor::prev() const {
    ASSERT(stack_.top()->prev_->data_->nodeType() == ASTNodeType::StmtSeq);
    auto seq = stack_.top()->prev_->data_.as<StmtSeqNode>();
    auto it = std::find_if(seq->stmts_.begin(), seq->stmts_.end(),
                           [&](const Stmt &s) { return s->id() == id(); });
    Cursor ret = *this;
    ret.pop(), ret.push(*(it - 1));
    return ret;
}

bool Cursor::hasNext() const {
    if (!stack_.top()->prev_.isValid()) {
        return false;
    }
    if (stack_.top()->prev_->data_->nodeType() != ASTNodeType::StmtSeq) {
        return false;
    }
    auto seq = stack_.top()->prev_->data_.as<StmtSeqNode>();
    auto it = std::find_if(seq->stmts_.rbegin(), seq->stmts_.rend(),
                           [&](const Stmt &s) { return s->id() == id(); });
    return it > seq->stmts_.rbegin();
}

Cursor Cursor::next() const {
    ASSERT(stack_.top()->prev_->data_->nodeType() == ASTNodeType::StmtSeq);
    auto seq = stack_.top()->prev_->data_.as<StmtSeqNode>();
    auto it = std::find_if(seq->stmts_.rbegin(), seq->stmts_.rend(),
                           [&](const Stmt &s) { return s->id() == id(); });
    Cursor ret = *this;
    ret.pop(), ret.push(*(it - 1));
    return ret;
}

bool Cursor::hasOuter() const {
    auto t = stack_.top();
    do {
        t = t->prev_;
        if (!t.isValid()) {
            return false;
        }
    } while (t->data_->nodeType() == ASTNodeType::StmtSeq ||
             t->data_->nodeType() == ASTNodeType::VarDef);
    return true;
}

Cursor Cursor::outer() const {
    auto ret = *this;
    do {
        ret.pop();
    } while (ret.node()->nodeType() == ASTNodeType::StmtSeq ||
             ret.node()->nodeType() == ASTNodeType::VarDef);
    return ret;
}

void VisitorWithCursor::visitStmt(
    const Stmt &op, const std::function<void(const Stmt &)> &visitNode) {
    cursor_.push(op);
    Visitor::visitStmt(op, visitNode);
    cursor_.pop();
}

void GetCursorById::visitStmt(
    const Stmt &op, const std::function<void(const Stmt &)> &visitNode) {
    if (!found_) {
        VisitorWithCursor::visitStmt(op, [&visitNode, this](const Stmt &_op) {
            visitNode(_op);
            if (_op->id() == id_) {
                result_ = cursor();
                ASSERT(result_.id() == id_);
                found_ = true;
            }
        });
    }
}

} // namespace ir
