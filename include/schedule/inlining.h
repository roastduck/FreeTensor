#ifndef INLINING_H
#define INLINING_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class MakeInlinePlaceholder : public Mutator {
    std::vector<Expr> indices_;

  public:
    MakeInlinePlaceholder(const std::vector<Expr> &indices)
        : indices_(indices) {}
    MakeInlinePlaceholder(const std::vector<SubTree<ExprNode>> &indices)
        : MakeInlinePlaceholder(
              std::vector<Expr>(indices.begin(), indices.end())) {}

  protected:
    Expr visitExpr(const Expr &op) override;
};

class ApplyInlinePlaceholder : public Mutator {
    std::vector<Expr> indices_;

  public:
    ApplyInlinePlaceholder(const std::vector<Expr> &indices)
        : indices_(indices) {}
    ApplyInlinePlaceholder(const std::vector<SubTree<ExprNode>> &indices)
        : ApplyInlinePlaceholder(
              std::vector<Expr>(indices.begin(), indices.end())) {}

  protected:
    Expr visit(const Var &op) override;
};

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
