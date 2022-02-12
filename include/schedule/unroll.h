#ifndef UNROLL_H
#define UNROLL_H

#include <mutator.h>

namespace ir {

/**
 * Mark a loop as to be unroll, and let a backend compiler deal with it
 */
class BackUnroll : public Mutator {
    ID loop_;
    bool done_ = false;

  public:
    BackUnroll(const ID &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visit(const For &op) override;
};

/**
 * Immediately unroll a loop just in the AST
 */
class ImmediateUnroll : public Mutator {
    ID loop_;
    bool done_ = false;

    std::string iter_;
    Expr begin_, step_;
    int curIter_;

  public:
    ImmediateUnroll(const ID &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
};

Stmt unroll(const Stmt &ast, const ID &loop, bool immediate);

} // namespace ir

#endif // UNROLL_H
