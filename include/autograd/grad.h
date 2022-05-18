#ifndef FREE_TENSOR_GRAD_H
#define FREE_TENSOR_GRAD_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

class PropagateRequire : public WithTypeInfer<SymbolTable<Visitor>> {
    typedef WithTypeInfer<SymbolTable<Visitor>> BaseClass;

    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<ID> affectedDefs_; // all VarDef IDs

    ID curTarget_; // VarDef ID of current var being written to

  public:
    PropagateRequire(const std::unordered_set<std::string> &_requires,
                     const std::unordered_set<std::string> &provides)
        : requires_(_requires), provides_(provides) {}

    const std::unordered_set<ID> &affectedDefs() const { return affectedDefs_; }

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

class ReplaceByTape : public Mutator {
    const SymbolTableInterface &symbolTable_;
    const std::unordered_map<ID, std::string> &tapeMap_;
    const std::unordered_map<ID, Expr> &versions_;
    Stmt parent_;

  public:
    ReplaceByTape(const SymbolTableInterface &symbolTable,
                  const std::unordered_map<ID, std::string> &tapeMap,
                  const std::unordered_map<ID, Expr> &versions,
                  const Stmt &parent)
        : symbolTable_(symbolTable), tapeMap_(tapeMap), versions_(versions),
          parent_(parent) {}

    Expr replaceForwardValue(const Expr &equLoad);

  protected:
    Expr visit(const Load &op) override;
};

class GradExpr : public Visitor {
    const std::unordered_map<std::string, std::string>
        &gradNames_;                           // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_; // x -> dy/dx
    const Expr &root_;
    Expr equLoad_;
    ReplaceByTape &replaceByTape_;
    std::vector<Stmt> appends_;

  public:
    GradExpr(ReplaceByTape &replaceByTape,
             const std::unordered_map<std::string, std::string> &gradNames,
             const Expr &root, const Expr &grad, const Expr &equLoad)
        : gradNames_(gradNames), root_(root), equLoad_(equLoad),
          replaceByTape_(replaceByTape) {
        gradExprs_[root] = grad;
    }

    const std::vector<Stmt> &appends() const { return appends_; }

  private:
    Expr replaceByLoadY(const Expr &op) {
        return HashComparator()(op, root_) ? equLoad_ : op;
    }

    Expr useForwardVal(const Expr &_op) {
        auto op = replaceByLoadY(_op);
        if (op == _op) {
            return replaceByTape_(op);
        } else {
            return replaceByTape_.replaceForwardValue(op);
        }
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

class Grad : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_set<std::string> &requires_;
    const std::unordered_set<std::string> &provides_;
    const std::unordered_set<ID> &tapes_;
    const std::unordered_set<ID> &affectedDefs_;
    const std::unordered_map<ID, std::string> &tapeMap_;
    const std::unordered_map<ID, Expr> &versions_;
    const std::unordered_map<ID, Expr> &totLens_;
    const std::unordered_set<Stmt> &notSingleWrite_;

    std::unordered_map<std::string, std::string> requireGrads_; // var name map
    std::unordered_map<std::string, std::string> provideGrads_; // var name map

    std::unordered_map<std::string, std::string> gradNames_; // x -> dy/dx
    std::unordered_set<std::string> taped_;
    std::unordered_map<Expr, Expr> equLoads_;
    std::unordered_map<std::string, std::unordered_set<Stmt>>
        recomputed_; // var name -> set{stmt}
    bool isRecompute_ = false;

  public:
    Grad(const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<ID> &tapes,
         const std::unordered_set<ID> &affectedDefs,
         const std::unordered_map<ID, std::string> &tapeMap,
         const std::unordered_map<ID, Expr> &versions,
         const std::unordered_map<ID, Expr> &totLens,
         const std::unordered_set<Stmt> &notSingleWrite)
        : requires_(_requires), provides_(provides), tapes_(tapes),
          affectedDefs_(affectedDefs), tapeMap_(tapeMap), versions_(versions),
          totLens_(totLens), notSingleWrite_(notSingleWrite) {}

    const std::unordered_map<std::string, std::string> &requireGrads() const {
        return requireGrads_;
    }
    const std::unordered_map<std::string, std::string> &provideGrads() const {
        return provideGrads_;
    }

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Reverse mode Auto differentiation
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
 * corresponding output names
 * )
 *
 * @{
 */
std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &_requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<ID> &tapes);

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &_requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<ID> &tapes);
/** @} */

enum class GradTapeMode : int { All, Nothing, NoReuseOnly };

/**
 * Reverse mode Auto differentiation
 *
 * @param op : Original AST
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
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
 * corresponding output names
 * )
 *
 * @{
 */
std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &_requires,
     const std::unordered_set<std::string> &provides,
     GradTapeMode tapeMode = GradTapeMode::NoReuseOnly);

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &_requires,
     const std::unordered_set<std::string> &provides,
     GradTapeMode tapeMode = GradTapeMode::NoReuseOnly);
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_GRAD_H
