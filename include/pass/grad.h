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

class ReplaceByTape : public Mutator {
    const std::unordered_map<Load, Expr> &loadMap_;

  public:
    ReplaceByTape(const std::unordered_map<Load, Expr> &loadMap)
        : loadMap_(loadMap) {}

  protected:
    Expr visit(const Load &op) override;
};

class GradExpr : public Visitor {
    const std::unordered_map<std::string, std::string>
        &gradNames_;                           // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_; // x -> dy/dx
    Expr root_, equLoad_;
    ReplaceByTape replaceByTape_;
    std::vector<Stmt> appends_;

  public:
    GradExpr(const std::unordered_map<Load, Expr> &loadMap,
             const std::unordered_map<std::string, std::string> &gradNames,
             const Expr &root, const Expr &grad, const Expr &equLoad)
        : gradNames_(gradNames), root_(root), equLoad_(equLoad),
          replaceByTape_(loadMap) {
        gradExprs_[root] = grad;
    }

    const std::vector<Stmt> &appends() const { return appends_; }

  private:
    Expr replaceByLoadY(const Expr &op) { return op == root_ ? equLoad_ : op; }

    Expr useForwardVal(const Expr &op) {
        return replaceByTape_(replaceByLoadY(op));
    }

  protected:
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

class Grad : public Mutator {
    const std::unordered_set<std::string> &requires_;
    const std::unordered_set<std::string> &provides_;
    const std::unordered_set<std::string> &tapes_;
    const std::unordered_set<std::string> &affectedDefs_;
    const std::unordered_map<Load, Expr> &loadMap_;
    std::unordered_set<std::string> isTape_;
    ReplaceByTape replaceByTape_;

    std::unordered_map<std::string, std::string> requireGrads_; // var name map
    std::unordered_map<std::string, std::string> provideGrads_; // var name map

    std::unordered_map<std::string, std::string> gradNames_; // x -> dy/dx
    std::unordered_map<std::string, VarDef> defs_;
    std::unordered_set<std::string> taped_;
    std::unordered_map<Expr, Expr> equLoads_;
    std::unordered_map<std::string, std::unordered_set<Stmt>>
        recomputed_; // var name -> set{stmt}
    bool isRecompute_ = false;

  public:
    Grad(const std::unordered_set<std::string> &requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<std::string> &tapes,
         const std::unordered_set<std::string> &affectedDefs,
         const std::unordered_map<std::string, std::string> &tapeMap,
         const std::unordered_map<Load, Expr> &loadMap)
        : requires_(requires), provides_(provides), tapes_(tapes),
          affectedDefs_(affectedDefs), loadMap_(loadMap),
          replaceByTape_(loadMap) {
        for (auto &&[oriDef, tapeVar] : tapeMap) {
            isTape_.insert(tapeVar);
        }
    }

    const std::unordered_map<std::string, std::string> &requireGrads() const {
        return requireGrads_;
    }
    const std::unordered_map<std::string, std::string> &provideGrads() const {
        return provideGrads_;
    }

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override { ASSERT(false); }
};

/**
 * Auto differentiation
 *
 * @param op : Original AST
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
 * @param tapes : VarDef IDs of intermediate variables that need to be stored in
 * the forward pass
 * @return : (
 *  Forward AST
 *  Backward AST,
 *  Mapping from names in requries to its gradient name,
 *  Mapping from names in provides to its gradient name
 *  Mapping from VarDef IDs of intermediate variables being stored to its
 * corresponding output names. Currently all output variables must be stored,
 * and should not be specified in tapes (TODO: allow not storing an output
 * variable)
 * )
 */
std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<std::string> &tapes);

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<std::string> &tapes);

enum class GradTapeMode : int { All, Nothing, NoReuseOnly };

/**
 * Auto differentiation
 *
 * @param op : Original AST
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
 * @param tapes : VarDef IDs of intermediate variables that need to be stored in
 * the forward pass
 * @param tapeMode : Mode of which intermediate variables should be stored. All:
 * store all variables including local scalars; None: store nothing;
 * NoReuseOnly: store variables that only hold one version of data, which means
 * we do not have to store each version of them in their history
 * @return : (
 *  Forward AST
 *  Backward AST,
 *  Mapping from names in requries to its gradient name,
 *  Mapping from names in provides to its gradient name,
 *  Mapping from VarDef IDs of intermediate variables being stored to its
 * corresponding output names. Currently all output variables must be stored,
 * and should not be specified in tapes (TODO: allow not storing an output
 * variable)
 * )
 */
std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     GradTapeMode tapeMode = GradTapeMode::NoReuseOnly);

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     GradTapeMode tapeMode = GradTapeMode::NoReuseOnly);

} // namespace ir

#endif // GRAD_H
