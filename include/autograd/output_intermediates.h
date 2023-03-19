#ifndef FREE_TENSOR_OUTPUT_INTERMEDIATES_H
#define FREE_TENSOR_OUTPUT_INTERMEDIATES_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <autograd/derivative.h>
#include <mutator.h>

namespace freetensor {

enum class OutputIntermediatesStage : int { Forward, Backward };

class OutputIntermediates : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<StmtOrExprID, Expr> &versions_;
    const std::unordered_map<ID, Expr> &totLens_;
    const std::unordered_set<ID> &trivials_;
    OutputIntermediatesStage stage_;
    std::string varSuffix_;

    std::unordered_map<ID, std::string> savedNames_;
    std::unordered_set<ID> insertedStmts_;
    std::unordered_map<ID, std::vector<Stmt>> toSave_;
    ID curStmt_;

  public:
    OutputIntermediates(const std::unordered_map<StmtOrExprID, Expr> &versions,
                        const std::unordered_map<ID, Expr> &totLens,
                        const std::unordered_set<ID> &trivials,
                        OutputIntermediatesStage stage,
                        const std::string &varSuffix)
        : versions_(versions), totLens_(totLens), trivials_(trivials),
          stage_(stage), varSuffix_(varSuffix) {}

    const auto &savedNames() const { return savedNames_; }
    const auto &insertedStmts() const { return insertedStmts_; }

  private:
    std::string savingName(const std::string &oldName) const;

  protected:
    using BaseClass::visit;
    Stmt visitStmt(const Stmt &stmt) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * Save all needed versions of some specified intermediate (AccessType::Cache)
 * variables in a program in larger tensors.
 *
 * Old intermediate variables are still preserved, but may be removed using
 * lowering passes or a `inline` schedule. Rationale: one intermediate (old)
 * element maps to multiple output (new) elements, so it is hard to determine
 * which element to load from, if directly loading from the output variable
 *
 * We save the variables both after where it is stored, AND before where it is
 * loaded.
 *
 * Saving before loading is necessary because there might be random accesses,
 * e.g.,
 *
 * ```
 * a[x] = 1
 * ... = a[x]
 * a[y] = 2
 * ... = a[z]  # z == x or z == y
 * ```
 *
 * will be transformed to
 *
 * ```
 * a[x] = 1
 * a.tape[0, x] = a[x]
 * a.tape[0, x] = a[x]
 * ... = a[x]
 * a[y] = 2
 * a.tape[1, y] = a[y]
 * a.tape[1, z] = a[z]
 * ... = a[z]
 * ```
 *
 * so that even `z != y`, we can get the correct Version 1 `a[z]` in `a.tape[1,
 * z]`.
 *
 * Saving after storing is also necessary because the gradient of y = f(x) may
 * better be a function of y, instead of a function of x.
 *
 * @params op : The program
 * @params intermediates : VarDef IDs of the intermediate variables
 * @param derivatives : Lazy derivative generators for each statement. We need
 * this information to determine which values are to be used in gradients
 * @params stage : If transforming the forward pass to save all versions of the
 * intermediates as `AccessType::Output` variables (i.e. tapes), set this to
 * `Forward`. If transforming the backward pass to save all versions of the
 * intermediates but still keeping them as `AccessType::Cache` variables, set
 * this to `Backward`
 * @params varSuffix : Name suffix to append to the new versioning tensors
 * @return : (
 *  The transformed program,
 *  Mapping from VarDef IDs of intermediate variables to the name of the new
 * saving tensors,
 *  Versions of each memory accesses, Total version counts of each
 * VarDef nodes,
 *  Set of all newly inserted statements,
 *  Mapping from tape_name to var name and explicit user versions marked via
 * mark_version
 * )
 */
std::tuple<Stmt, std::unordered_map<ID, std::string>,
           std::unordered_map<StmtOrExprID, Expr>, std::unordered_map<ID, Expr>,
           std::unordered_set<ID>,
           std::unordered_map<std::string, std::pair<std::string, Expr>>>
outputIntermediates(
    const Stmt &op, const std::unordered_set<ID> &intermediates,
    const std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives,
    OutputIntermediatesStage stage = OutputIntermediatesStage::Forward,
    const std::string &varSuffix = ".tape");

} // namespace freetensor

#endif // FREE_TENSOR_OUTPUT_INTERMEDIATES_H
