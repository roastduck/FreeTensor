#ifndef CURSOR_H
#define CURSOR_H

#include <functional>
#include <vector>

#include <except.h>
#include <stack.h>
#include <stmt.h>
#include <visitor.h>

namespace ir {

class Cursor {
    friend class VisitorWithCursor;

    Stack<Stmt> stack_;

  private:
    void push(const Stmt &op) { stack_.push(op); }
    void pop() { stack_.pop(); }

  public:
    Cursor() {}

    const Stmt &node() const { return stack_.top()->data_; }

    const std::string &id() const { return node()->id(); }

    Stmt getParentById(const std::string &id) const;

    bool isBefore(const Cursor &other) const;
    bool isOuter(const Cursor &other) const;

    // The following functions returns a new cursor pointing to a nearby
    // position

    /// The previous statement in the same StmtSeq
    Cursor prev() const;
    bool hasPrev() const;

    /// The next statement in the same StmtSeq
    Cursor next() const;
    bool hasNext() const;

    /// The parent in the AST tree
    Cursor outer() const;
    bool hasOuter() const;
};

class VisitorWithCursor : public Visitor {
    Cursor cursor_;

  protected:
    void visitStmt(const Stmt &op,
                   const std::function<void(const Stmt &)> &visitNode) override;

    const Cursor &cursor() const { return cursor_; }
};

class GetCursorById : public VisitorWithCursor {
    std::string id_;
    Cursor result_;
    bool found_ = false;

  public:
    GetCursorById(const std::string &id) : id_(id) {}

    const Cursor &result() const { return result_; }
    bool found() const { return found_; }

  protected:
    void visitStmt(const Stmt &op,
                   const std::function<void(const Stmt &)> &visitNode) override;
};

inline Cursor getCursorById(const Stmt &ast, const std::string &id) {
    GetCursorById visitor(id);
    visitor(ast);
    if (!visitor.found()) {
        throw InvalidSchedule("Statement " + id + " not found");
    }
    return visitor.result();
}

class GetCursorByFilter : public VisitorWithCursor {
    const std::function<bool(const Cursor &)> &filter_;
    std::vector<Cursor> results_;

  public:
    GetCursorByFilter(const std::function<bool(const Cursor &)> &filter)
        : filter_(filter) {}
    const std::vector<Cursor> &results() const { return results_; }

  protected:
    void visitStmt(const Stmt &op,
                   const std::function<void(const Stmt &)> &visitNode) override;
};

inline std::vector<Cursor>
getCursorByFilter(const Stmt &ast,
                  const std::function<bool(const Cursor &)> &filter) {
    GetCursorByFilter visitor(filter);
    visitor(ast);
    return visitor.results();
}

} // namespace ir

#endif // CURSOR_H
