#ifndef CURSOR_H
#define CURSOR_H

#include <functional>
#include <vector>

#include <except.h>
#include <mutator.h>
#include <stack.h>
#include <stmt.h>
#include <visitor.h>

namespace ir {

class Cursor {
    template <class T> friend class WithCursor;
    friend class GetCursorById;
    friend class GetCursorByFilter;

    Stack<Stmt> stack_;

  private:
    void push(const Stmt &op) { stack_.push(op); }
    void pop() { stack_.pop(); }

  public:
    Cursor() {}

    bool isValid() const { return !stack_.empty(); }
    size_t depth() const { return stack_.size(); }

    const Stmt &node() const { return stack_.top()->data_; }
    ID id() const { return node()->id(); }
    ASTNodeType nodeType() const { return node()->nodeType(); }

    Stmt getParentById(const ID &id) const;

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

    /// The parent which is not StmtSeq or VarDef in the AST tree
    Cursor outerCtrlFlow() const;
    bool hasOuterCtrlFlow() const;

    /// Lowest common ancestor
    friend Cursor lca(const Cursor &lhs, const Cursor &rhs);
};

} // namespace ir

#endif // CURSOR_H
