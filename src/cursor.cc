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
    auto lMode = mode_, rMode = other.mode_;
    while (l.size() > r.size()) {
        l.pop(), lMode = CursorMode::Range;
    }
    while (r.size() > l.size()) {
        r.pop(), rMode = CursorMode::Range;
    }
    for (; !l.empty(); l.pop(), r.pop()) {
        if (l.top()->data_->id() == r.top()->data_->id()) {
            switch (lMode) {
            case CursorMode::All:
            case CursorMode::End:
                return false;
            case CursorMode::Range:
                return rMode == CursorMode::End;
            case CursorMode::Begin:
                return rMode == CursorMode::Range || rMode == CursorMode::End;
            }
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
    } while (ret.top()->nodeType() == ASTNodeType::StmtSeq ||
             ret.top()->nodeType() == ASTNodeType::VarDef);
    return ret;
}

void VisitorWithCursor::operator()(const AST &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Any:
    case ASTNodeType::StmtSeq:
    case ASTNodeType::VarDef:
    case ASTNodeType::Store:
    case ASTNodeType::ReduceTo:
    case ASTNodeType::For:
    case ASTNodeType::If:
    case ASTNodeType::Assert:
    case ASTNodeType::Eval:
        cursor_.push(op.as<StmtNode>());
        Visitor::operator()(op);
        cursor_.pop();
        break;
    default:
        Visitor::operator()(op);
    }
}

} // namespace ir

