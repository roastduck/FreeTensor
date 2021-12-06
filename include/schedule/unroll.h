#ifndef UNROLL_H
#define UNROLL_H

#include <mutator.h>

namespace ir {

/**
 * Mark a loop as to be unroll, and let a backend compiler deal with it
 */
class BackUnroll : public Mutator {
    std::string loop_;
    bool done_ = false;

  public:
    BackUnroll(const std::string &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visit(const For &op) override;
};

/**
 * Immediately unroll a loop just in the AST
 */
class ImmediateUnroll : public Mutator {
    std::string loop_;
    bool done_ = false;

    std::string iter_;
    Expr begin_, step_;
    int curIter_;

  public:
    ImmediateUnroll(const std::string &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Expr visit(const Var &op) override;
    Stmt visit(const For &op) override;
};

Stmt unroll(const Stmt &ast, const std::string &loop, bool immediate);

} // namespace ir

#endif // UNROLL_H
