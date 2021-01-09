#ifndef CURSOR_H
#define CURSOR_H

#include <vector>

#include <except.h>
#include <stack.h>
#include <stmt.h>
#include <visitor.h>

namespace ir {

enum class CursorMode : int { All, Range, Begin, End };

class Cursor {
    friend class VisitorWithCursor;

    Stack<Stmt> stack_;
    CursorMode mode_ = CursorMode::All;

  private:
    void push(const Stmt &op) { stack_.push(op); }
    void pop() { stack_.pop(); }

  public:
    Cursor() {}

    void setMode(CursorMode mode) { mode_ = mode; }

    const Stmt &top() const { return stack_.top()->data_; }

    const std::string &id() const { return top()->id(); }

    Stmt getParentById(const std::string &id) const;

    bool isBefore(const Cursor &other) const;

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

  public:
    virtual void operator()(const AST &op) override;

  protected:
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

  private:
    template <class T> void doVisit(const T &op) {
        Visitor::visit(op);
        if (op->id() == id_) {
            result_ = cursor();
            found_ = true;
        }
    }

  protected:
    virtual void visit(const Any &op) override { doVisit(op); }
    virtual void visit(const VarDef &op) override { doVisit(op); }
    virtual void visit(const Store &op) override { doVisit(op); }
    virtual void visit(const AddTo &op) override { doVisit(op); }
    virtual void visit(const For &op) override { doVisit(op); }
    virtual void visit(const If &op) override { doVisit(op); }
    virtual void visit(const Assert &op) override { doVisit(op); }
};

inline Cursor getCursorById(const Stmt &ast, const std::string &id) {
    GetCursorById visitor(id);
    visitor(ast);
    if (!visitor.found()) {
        throw InvalidSchedule("Statement " + id + " not found");
    }
    return visitor.result();
}

} // namespace ir

#endif // CURSOR_H
