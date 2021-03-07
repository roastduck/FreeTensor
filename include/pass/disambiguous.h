#ifndef DISAMBIGUOUS_H
#define DISAMBIGUOUS_H

#include <mutator.h>

namespace ir {

/**
 * Make there will be no shared node in a AST
 *
 * If any pass uses node addresses as keys, perform Disambiguous first
 */
class Disambiguous : public Mutator {
  public:
    Stmt visitStmt(const Stmt &op,
                    const std::function<Stmt(const Stmt &)> &visitNode) override;

    Expr visitExpr(const Expr &op,
                    const std::function<Expr(const Expr &)> &visitNode) override;
};

inline Stmt disambiguous(const Stmt &op) { return Disambiguous()(op); }
inline Expr disambiguous(const Expr &op) { return Disambiguous()(op); }

} // namespace ir

#endif // DISAMBIGUOUS_H
