#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class ReplaceVar : public Mutator {
    std::string from_;
    Expr to_;

  public:
    ReplaceVar(const std::string &from, const Expr &to)
        : from_(from), to_(to) {}

  protected:
    Expr visit(const Var &op) override;
};

class Grad : public Visitor {
    std::unordered_map<std::string, std::string> gradNames_; // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_;               // x -> dy/dx
    std::unordered_map<Stmt, Stmt> gradStmts_;               // x -> dy/dx
    std::unordered_map<Stmt, Stmt> oriStmts_;                // x -> x
    std::vector<Stmt> appends_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;

  public:
    Grad(const std::unordered_map<std::string, std::string> &gradNames)
        : gradNames_(gradNames) {}

    Stmt grad(const Stmt &root) const { return gradStmts_.at(root); }

  protected:
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const RealDiv &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const IfExpr &op) override;
};

inline Stmt
grad(const Stmt &op,
     const std::unordered_map<std::string, std::string> &gradNames) {
    Grad visitor(gradNames);
    visitor(op);
    return visitor.grad(op);
}

inline Func
grad(const Func &func,
     const std::unordered_map<std::string, std::string> &gradNames) {
    std::vector<std::string> params;
    for (auto &&param : func->params_) {
        params.emplace_back(param);
        if (gradNames.count(param)) {
            params.emplace_back(gradNames.at(param));
        }
    }
    return makeFunc(func->name_, std::move(params),
                    grad(func->body_, gradNames), nullptr);
}

} // namespace ir

#endif // GRAD_H
