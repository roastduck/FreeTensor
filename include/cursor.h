#ifndef CURSOR_H
#define CURSOR_H

#include <vector>

#include <stmt.h>

namespace ir {

class Cursor {
    std::vector<Stmt> stack_;

  public:
    Cursor() {}
    Cursor(const Stmt &op) : stack_{{op}} {}

    const Stmt &top() const { return stack_.back(); }

    const std::string &id() const { return top()->id(); }

    Stmt getParentById(const std::string &id) const;

    void enter(const Stmt &op) { stack_.emplace_back(op); }

    void leave() { stack_.pop_back(); }
};

} // namespace ir

#endif // CURSOR_H
