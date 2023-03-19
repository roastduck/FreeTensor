#ifndef FREE_TENSOR_REPLACE_BY_SAVED_H
#define FREE_TENSOR_REPLACE_BY_SAVED_H

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

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
    ID rootStmtID_;
    Store alreadyStored_;
    bool isGrad_ = false;

  public:
    ReplaceBySaved(const SymbolTableInterface &symbolTable,
                   const std::unordered_map<ID, std::string> &intermediatesMap,
                   const std::unordered_map<StmtOrExprID, Expr> &versions,
                   const ID &rootStmtID, const Store alreadyStored_ = nullptr)
        : symbolTable_(symbolTable), intermediatesMap_(intermediatesMap),
          versions_(versions), rootStmtID_(rootStmtID),
          alreadyStored_(alreadyStored_) {}

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

} // namespace freetensor

#endif // FREE_TENSOR_REPLACE_BY_SAVED_H
