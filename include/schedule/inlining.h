#ifndef INLINING_H
#define INLINING_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class MakeInline : public Mutator {
    std::string def_, var_;
    const std::unordered_map<Load, Expr> &replace_;

  public:
    MakeInline(const std::string &def,
               const std::unordered_map<Load, Expr> &replace)
        : def_(def), replace_(replace) {}

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

} // namespace ir

#endif // INLINING_H
