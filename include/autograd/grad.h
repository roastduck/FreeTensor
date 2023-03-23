#ifndef FREE_TENSOR_GRAD_H
#define FREE_TENSOR_GRAD_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <autograd/derivative.h>
#include <autograd/invert_stmts.h>
#include <autograd/replace_by_saved.h>
#include <autograd/user_grad.h>
#include <func.h>
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

    std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives_; // Mutable for lazy operations
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
    const std::unordered_map<ID, InversionInfo> &inverseStmts_;
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
    Grad(std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
             &derivatives,
         const std::unordered_set<std::string> &_requires,
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
         const std::unordered_map<ID, InversionInfo> &inverseStmts,
         const std::vector<RangeToUserGrad> &userGrads)
        : derivatives_(derivatives), requires_(_requires), provides_(provides),
          tapes_(tapes), affectedDefs_(affectedDefs),
          intermediatesMap_(intermediatesMap), versions_(versions),
          userVersions_(userVersions), totLens_(totLens),
          saveLocalStmts_(saveLocalStmts), notSingleWrite_(notSingleWrite),
          inverseStmts_(inverseStmts), userGrads_(userGrads) {}

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
 * @param invert: If set to true, it can reduce the amount of recomputation or
 * taping required. However, this may result in a loss of precision for
 * floating-point numbers. Defaults to true.
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
         const std::unordered_set<ID> &tapes, bool invert = true,
         const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                const std::unordered_set<ID> &tapes, bool tapeInClosure = true,
                bool invert = true,
                const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   const std::unordered_set<ID> &tapes,
                   bool tapeInClosure = true, bool invert = true,
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
 * @param invert: If set to true, it can reduce the amount of recomputation or
 * taping required. However, this may result in a loss of precision for
 * floating-point numbers. Defaults to true.
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
         GradTapeMode tapeMode = GradTapeMode::NoReuseOnly, bool invert = true,
         const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                GradTapeMode tapeMode = GradTapeMode::NoReuseOnly,
                bool tapeInClosure = true, bool invert = true,
                const std::vector<StmtSetToUserGrad> &userGrads = {});

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   GradTapeMode tapeMode = GradTapeMode::NoReuseOnly,
                   bool tapeInClosure = true, bool invert = true,
                   const std::vector<StmtSetToUserGrad> &userGrads = {});
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_GRAD_H
