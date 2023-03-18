#ifndef FREE_TENSOR_GRAD_H
#define FREE_TENSOR_GRAD_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <autograd/user_grad.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

/**
 * Determine what variables we need to compute gradient for (propagete from
 * inputs to outputs)
 */
class PropagateRequires : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<ID> affectedDefs_; // all VarDef IDs

    ID curTarget_; // VarDef ID of current var being written to

  public:
    PropagateRequires(const std::unordered_set<std::string> &_requires,
                      const std::unordered_set<std::string> &provides)
        : requires_(_requires), provides_(provides) {}

    const std::unordered_set<ID> &affectedDefs() const { return affectedDefs_; }

    static std::unordered_set<ID>
    propagateUntilConverge(const Stmt &op,
                           const std::unordered_set<std::string> &_requires,
                           const std::unordered_set<std::string> &provides);

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

/**
 * Determine what variables we need to compute gradient for (propagete from
 * outputs to inputs)
 */
class PropagateProvides : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    const std::unordered_set<std::string> &requires_; // input var names
    const std::unordered_set<std::string> &provides_; // output var names

    std::unordered_set<ID> affectedDefs_; // all VarDef IDs

    ID curTarget_; // VarDef ID of current var being written to

  public:
    PropagateProvides(const std::unordered_set<std::string> &_requires,
                      const std::unordered_set<std::string> &provides)
        : requires_(_requires), provides_(provides) {}

    const std::unordered_set<ID> &affectedDefs() const { return affectedDefs_; }

    static std::unordered_set<ID>
    propagateUntilConverge(const Stmt &op,
                           const std::unordered_set<std::string> &_requires,
                           const std::unordered_set<std::string> &provides);

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
};

/**
 * Instead of directly using the variables from the original program, use taped
 * or recomputed variables as replacements
 *
 * (For gradient and recomputation) For each Load node, if it has a taped or
 * recomputed counterpart, replace it with new Load node that loads the taped or
 * recomputed version. In this case, versions of the Load node is used
 *
 * (Only for gradient only) For each sub-expression that matches
 * `alreadyStored`, it is re-loaded without computation. If the stored variable
 * has a taped or recomputed counterpart, also replace it. In this case,
 * versions of the Store node is used, which is one version later than that of a
 * Load node. This is for gradient only, or otherwise we will "re"-load what we
 * have not even computed
 */
class ReplaceBySaved : public Mutator {
    const SymbolTableInterface &symbolTable_;
    const std::unordered_map<ID, std::string> &intermediatesMap_;
    const std::unordered_map<StmtOrExprID, Expr> &versions_;
    Stmt parent_;
    Store alreadyStored_;
    bool isGrad_ = false;

  public:
    ReplaceBySaved(const SymbolTableInterface &symbolTable,
                   const std::unordered_map<ID, std::string> &intermediatesMap,
                   const std::unordered_map<StmtOrExprID, Expr> &versions,
                   const Stmt &parent, const Store alreadyStored_ = nullptr)
        : symbolTable_(symbolTable), intermediatesMap_(intermediatesMap),
          versions_(versions), parent_(parent), alreadyStored_(alreadyStored_) {
    }

    // Replace recomputing expressions
    auto recomp(const auto &op) {
        isGrad_ = false;
        return (*this)(op);
    }

    // Replace gradient expressions
    auto grad(const auto &op) {
        isGrad_ = true;
        return (*this)(op);
    }

  private:
    // Disabled. Use `ReplcaeBySaved::recomp` or `RepalceBySaved::grad` instaed
    using Mutator::operator();

  protected:
    Expr visitExpr(const Expr &expr) override;
    Expr visit(const Load &op) override;
};

class ReplaceLoadAtVersion : public Mutator {
    const SymbolTableInterface &symbolTable_;
    const std::unordered_map<ID, std::string> &intermediatesMap_;
    const std::unordered_map<std::string, std::pair<std::string, Expr>>
        &userVersions_;

  public:
    ReplaceLoadAtVersion(
        const SymbolTableInterface &symbolTable,
        const std::unordered_map<ID, std::string> &intermediatesMap,
        const std::unordered_map<std::string, std::pair<std::string, Expr>>
            &userVersions)
        : symbolTable_(symbolTable), intermediatesMap_(intermediatesMap),
          userVersions_(userVersions) {}

  protected:
    Expr visit(const LoadAtVersion &op) override;
};

class GradExpr : public Visitor {
    const std::unordered_map<std::string, std::string>
        &gradNames_;                           // x -> dy/dx
    std::unordered_map<Expr, Expr> gradExprs_; // x -> dy/dx
    const Expr &root_;
    ReplaceBySaved &replaceByTape_;
    std::vector<Stmt> appends_;

  public:
    GradExpr(ReplaceBySaved &replaceByTape,
             const std::unordered_map<std::string, std::string> &gradNames,
             const Expr &root, const Expr &grad)
        : gradNames_(gradNames), root_(root), replaceByTape_(replaceByTape) {
        gradExprs_[root] = grad;
    }

    const std::vector<Stmt> &appends() const { return appends_; }

  private:
    Expr useForwardVal(const Expr &op) { return replaceByTape_.grad(op); }

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
    void visit(const Ln &op) override;
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Intrinsic &op) override;
};

template <class BaseClass> class RenewIDs : public BaseClass {
  protected:
    Stmt visitStmt(const Stmt &s) override {
        auto ret = BaseClass::visitStmt(s);
        ret->setId();
        return ret;
    }
};

class Grad : public RenewIDs<SymbolTable<Mutator>> {
    // Because a statement can be both recomputed and computed for gradient, and
    // we may even recompute a statement several times, all IDs must be renewed,
    // even for recomputation
    typedef RenewIDs<SymbolTable<Mutator>> BaseClass;

    const std::unordered_set<std::string> &requires_;
    const std::unordered_set<std::string> &provides_;
    const std::unordered_set<ID> &tapes_;
    const std::unordered_set<ID> &affectedDefs_;
    const std::unordered_map<ID, std::string>
        &intermediatesMap_; // All saved variables, including in forward stage
                            // (tapes) and backward stage (during recomputation)
    const std::unordered_map<StmtOrExprID, Expr> &versions_;
    const std::unordered_map<std::string, std::pair<std::string, Expr>>
        &userVersions_;
    const std::unordered_map<ID, Expr> &totLens_;
    const std::unordered_set<ID> &saveLocalStmts_;
    const std::unordered_set<Stmt> &notSingleWrite_;
    std::vector<RangeToUserGrad> userGrads_; // mutable

    std::unordered_map<std::string, std::string> requireGrads_; // var name map
    std::unordered_map<std::string, std::string> provideGrads_; // var name map

    std::unordered_map<std::string, std::string> gradNames_; // x -> dy/dx
    std::unordered_set<std::string> taped_;
    std::unordered_map<Expr, Expr> equLoads_;
    std::unordered_map<std::string, std::unordered_set<Stmt>>
        recomputed_; // var name -> set{stmt}
    bool isRecompute_ = false;

    std::optional<RangeToUserGrad> userGradOpen_;
    ID userGradInsertPos_;

  private:
    /**
     * Get an appropriate ReplaceBySvaed Mutator to replace variables by saved
     * all-version variables
     *
     * - For recomputation, we need to replace only by tapes
     * - For gradient, we need to replace both by tapes and tensors saved during
     * recomputation
     */
    ReplaceBySaved getReplacer(const Stmt &stmt,
                               const Store &alreadyStored = nullptr) const;

    Stmt doVisitStmt(const Stmt &s);

  public:
    Grad(const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<ID> &tapes,
         const std::unordered_set<ID> &affectedDefs,
         const std::unordered_map<ID, std::string> &intermediatesMap,
         const std::unordered_map<StmtOrExprID, Expr> &versions,
         const std::unordered_map<std::string, std::pair<std::string, Expr>>
             &userVersions,
         const std::unordered_map<ID, Expr> &totLens,
         const std::unordered_set<ID> &saveLocalStmts,
         const std::unordered_set<Stmt> &notSingleWrite,
         const std::vector<RangeToUserGrad> &userGrads)
        : requires_(_requires), provides_(provides), tapes_(tapes),
          affectedDefs_(affectedDefs), intermediatesMap_(intermediatesMap),
          versions_(versions), userVersions_(userVersions), totLens_(totLens),
          saveLocalStmts_(saveLocalStmts), notSingleWrite_(notSingleWrite),
          userGrads_(userGrads) {}

    const std::unordered_map<std::string, std::string> &requireGrads() const {
        return requireGrads_;
    }
    const std::unordered_map<std::string, std::string> &provideGrads() const {
        return provideGrads_;
    }

  protected:
    Stmt visitStmt(const Stmt &s) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Reverse mode automatic differentiation
 *
 * @param op : (For `gradBody`) Original AST
 * @param op : (For `gradFuncInplace` and `gradFuncOutOfPlace`) Original
 * function
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
 * @param tapes : VarDef IDs of intermediate variables that need to be stored in
 * the forward pass
 * @param tapeInClosure : (For `gradFuncInplace` and `gradFuncOutOfPlace`) True
 * to pass taped tensors from the forward function to the backward function in
 * implicit I/O parameters, i.e. in closure. False to pass these tensors as
 * explicit I/O parameters. Default to true
 * @param userGrads : For custom gradients. Each `StmtSetToUserGrad` item in the
 * list specifies a statement range in the original program, which should be
 * replaced by a backward statement
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
gradBody(const Stmt &op, const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<ID> &tapes,
         const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                const std::unordered_set<ID> &tapes, bool tapeInClosure = true,
                const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   const std::unordered_set<ID> &tapes,
                   bool tapeInClosure = true,
                   const std::vector<StmtSetToUserGrad> &userGrads = {});
/** @} */

enum class GradTapeMode : int { All, Nothing, NoReuseOnly };

/**
 * Reverse mode automatic differentiation
 *
 * @param op : (For `gradBody`) Original AST
 * @param op : (For `gradFuncInplace` and `gradFuncOutOfPlace`) Original
 * function
 * @param requires : Name of input variables that need gradients
 * @param provides : Name of output variables whose gradients are known
 * @param tapeMode : Mode of which intermediate variables should be stored. All:
 * store all variables including local scalars; None: store nothing;
 * NoReuseOnly: store variables that only hold one version of data, which means
 * we do not have to store each version of them in their history
 * @param tapeInClosure : (For `gradFuncInplace` and `gradFuncOutOfPlace`) True
 * to pass taped tensors from the forward function to the backward function in
 * implicit I/O parameters, i.e. in closure. False to pass these tensors as
 * explicit I/O parameters. Default to true
 * @param userGrads : For custom gradients. Each `StmtSetToUserGrad` item in the
 * list specifies a statement range in the original program, which should be
 * replaced by a backward statement
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
gradBody(const Stmt &op, const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         GradTapeMode tapeMode = GradTapeMode::NoReuseOnly,
         const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                GradTapeMode tapeMode = GradTapeMode::NoReuseOnly,
                bool tapeInClosure = true,
                const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   GradTapeMode tapeMode = GradTapeMode::NoReuseOnly,
                   bool tapeInClosure = true,
                   const std::vector<StmtSetToUserGrad> &userGrads = {});
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_GRAD_H
