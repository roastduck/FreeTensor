#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class PropagateRequire : public Visitor {
    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names
    TypeInfer typeInfer_;

    std::unordered_set<std::string> affectedDefs_; // all VarDef IDs

    std::unordered_map<std::string, VarDef> defs_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    std::string curTarget_; // VarDef ID of current var being written to

  public:
    PropagateRequire(const std::unordered_set<std::string> &requires,
                     const std::unordered_set<std::string> &provides)
        : requires_(requires), provides_(provides), typeInfer_(&buffers_) {}

    const std::unordered_set<std::string> &affectedDefs() const {
        return affectedDefs_;
    }

  private:
    DataType dtype(const Expr &op);

  protected:
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

class ReplaceByTape : public Mutator {
    const std::unordered_map<std::string, VarDef> &defs_;
    const std::unordered_map<std::string, std::string> &tapeMap_;
    const std::unordered_map<AST, Expr> &versions_;

  public:
    ReplaceByTape(const std::unordered_map<std::string, VarDef> &defs,
                  const std::unordered_map<std::string, std::string> &tapeMap,
                  const std::unordered_map<AST, Expr> &versions)
        : defs_(defs), tapeMap_(tapeMap), versions_(versions) {}

  protected:
    Expr visit(const Load &op) override;
};

class GradExpr : public Visitor {
    const std::unordered_map<std::string, std::string>
        &gradNames_;                           // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_; // x -> dy/dx
    uint64_t rootHash_;
    Expr equLoad_;
    ReplaceByTape &replaceByTape_;
    std::vector<Stmt> appends_;

  public:
    GradExpr(ReplaceByTape &replaceByTape,
             const std::unordered_map<std::string, std::string> &gradNames,
             const Expr &root, const Expr &grad, const Expr &equLoad)
        : gradNames_(gradNames), rootHash_(getHash(root)), equLoad_(equLoad),
          replaceByTape_(replaceByTape) {
        gradExprs_[root] = grad;
    }

    const std::vector<Stmt> &appends() const { return appends_; }

  private:
    Expr replaceByLoadY(const Expr &op) {
        return getHash(op) == rootHash_ ? equLoad_ : op;
    }

    Expr useForwardVal(const Expr &op) {
        return replaceByLoadY(replaceByTape_(op));
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
    void visit(const Sigmoid &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
};

class Grad : public Mutator {
    const std::unordered_set<std::string> &requires_;
    const std::unordered_set<std::string> &provides_;
    const std::unordered_set<std::string> &tapes_;
    const std::unordered_set<std::string> &affectedDefs_;
    const std::unordered_map<std::string, std::string> &tapeMap_;
    const std::unordered_map<AST, Expr> &versions_;
    const std::unordered_map<std::string, Expr> &totLens_;
    const std::unordered_set<Stmt> &notSingleWrite_;
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
         const std::unordered_map<AST, Expr> &versions,
         const std::unordered_map<std::string, Expr> &totLens,
         const std::unordered_set<Stmt> &notSingleWrite)
        : requires_(requires), provides_(provides), tapes_(tapes),
          affectedDefs_(affectedDefs), tapeMap_(tapeMap), versions_(versions),
          totLens_(totLens), notSingleWrite_(notSingleWrite),
          replaceByTape_(defs_, tapeMap_, versions) {}

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
    Stmt visit(const ReduceTo &op) override;
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
