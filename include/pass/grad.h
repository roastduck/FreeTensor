#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>
#include <unordered_set>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class PropagateRequire : public Visitor {
    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<std::string> affectedDefs_; // all VarDef IDs

    std::unordered_map<std::string, VarDef> defs_;
    std::string curTarget_; // VarDef ID of current var being written to

  public:
    PropagateRequire(const std::unordered_set<std::string> &requires,
                     const std::unordered_set<std::string> &provides)
        : requires_(requires), provides_(provides) {}

    const std::unordered_set<std::string> &affectedDefs() const {
        return affectedDefs_;
    }

  protected:
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

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
    const std::unordered_set<std::string> &requires_;
    const std::unordered_set<std::string> &provides_;
    const std::unordered_set<std::string> &affectedDefs_;

    std::unordered_map<std::string, std::string> requireGrads_; // var name map
    std::unordered_map<std::string, std::string> provideGrads_; // var name map

    std::unordered_map<std::string, std::string> gradNames_; // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_;               // x -> dy/dx
    std::unordered_map<Stmt, Stmt> gradStmts_;               // x -> dy/dx
    std::unordered_map<Stmt, Stmt> oriStmts_;                // x -> x
    std::vector<Stmt> appends_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;

  public:
    Grad(const std::unordered_set<std::string> &requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<std::string> &affectedDefs)
        : requires_(requires), provides_(provides),
          affectedDefs_(affectedDefs) {}

    Stmt grad(const Stmt &root) const { return gradStmts_.at(root); }

    const std::unordered_map<std::string, std::string> &requireGrads() const {
        return requireGrads_;
    }
    const std::unordered_map<std::string, std::string> &provideGrads() const {
        return provideGrads_;
    }

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
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Square &op) override;
    void visit(const Abs &op) override;
};

/**
 * Auto differentiation
 *
 * @param op : Original AST
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
 * @return : (
 *  Backward AST,
 *  Mapping from names in requries to its gradient name,
 *  Mapping from names in provides to its gradient name
 * )
 */
std::tuple<Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides);

std::tuple<Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides);

} // namespace ir

#endif // GRAD_H
