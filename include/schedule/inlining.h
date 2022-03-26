#ifndef INLINING_H
#define INLINING_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class MakeInline : public Mutator {
    ID def_;
    std::string var_;
    const std::unordered_map<Load, Expr> &replace_;

  public:
    MakeInline(const ID &def, const std::unordered_map<Load, Expr> &replace)
        : def_(def), replace_(replace) {}

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt inlining(const Stmt &ast, const ID &def);

} // namespace ir

#endif // INLINING_H
