#ifndef FREE_TENSOR_COMP_ACCESS_BOUND_H
#define FREE_TENSOR_COMP_ACCESS_BOUND_H

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <math/bounds.h>
#include <visitor.h>

namespace freetensor {

struct AccessBound {
    std::vector<Expr> lower_; /// lower_bound(access)
    std::vector<Expr> upper_; /// upper_bound(access)
    std::vector<Expr> len_;   /// upper_bound(access) - lower_bound(access) + 1
    Expr cond_;               /// Conditions surrounding the accesses
    // TODO: Ideally, len_ should be upper_bound(access_i - access_j) + 1, which
    // supports shrinking a skewed variable, instead of upper_bound(access) -
    // lower_bound(aceess). However, it brins too much burden on pass/simplify,
    // so we do not choose it for now
};

typedef int CompAccessBoundMode;
const CompAccessBoundMode COMP_ACCESS_BOUND_READ = 0x1;
const CompAccessBoundMode COMP_ACCESS_BOUND_WRITE = 0x2;
const CompAccessBoundMode COMP_ACCESS_BOUND_ALL =
    COMP_ACCESS_BOUND_READ | COMP_ACCESS_BOUND_WRITE;

class FindMemType : public Visitor {
    ID varDefId_;
    MemType mtype_;

  public:
    FindMemType(const ID &varDefId) : varDefId_(varDefId) {}

    MemType mtype() const { return mtype_; }

  protected:
    void visit(const VarDef &op) override;
};

class CompAccessBound : public CompTransientBounds<SymbolTable<Visitor>> {
    typedef CompTransientBounds<SymbolTable<Visitor>> BaseClass;

  public:
    struct Access {
        std::vector<Expr> indices_, conds_;
        std::vector<Ref<CompUniqueBounds::Bound>> bounds_;

        Access(CompUniqueBounds &unique, const std::vector<Expr> &indices,
               const std::vector<Expr> &conds,
               const std::unordered_set<std::string> &names)
            : indices_(indices), conds_(conds) {
            for (auto &&idx : indices) {
                bounds_.emplace_back(
                    unique.getBound(idx)->restrictScope(names));
            }
        }

        Access(CompUniqueBounds &unique, const std::vector<Expr> &indices,
               const std::vector<Expr> &conds)
            : indices_(indices), conds_(conds) {
            for (auto &&idx : indices) {
                bounds_.emplace_back(unique.getBound(idx));
            }
        }
    };

  private:
    Ref<CompUniqueBounds> unique_;

    // The variable to compute
    ID varDefId_;
    std::string var_;
    MemType mtype_;

    // each access to the specific variable
    std::vector<Access> access_;

    // all defined name in the scope
    std::unordered_set<std::string> defs_;
    std::unordered_map<std::string, std::unordered_set<std::string>>
        defsAtVarDef_;

    CompAccessBoundMode mode_;
    bool includeTrivialBound_;

    ID filterSubTree_;
    bool filtered_ = false;

    AccessBound result_;

  public:
    CompAccessBound(const ID &varDefId, MemType mtype,
                    CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL,
                    bool includeTrivialBound = true,
                    const ID &filterSubTree = ID())
        : varDefId_(varDefId), mtype_(mtype), mode_(mode),
          includeTrivialBound_(includeTrivialBound),
          filterSubTree_(filterSubTree) {
        if (!filterSubTree_.isValid()) {
            filtered_ = true;
        }
    }

    const AccessBound &result() const { return result_; }

  protected:
    using BaseClass::visit;
    void visitStmt(const Stmt &stmt) override;
    void visit(const VarDef &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const For &op) override;
};

/**
 * Compute the bound of all indices indexing a particular variable
 *
 * @param op : AST to be analyzed
 * @param varDefId : ID of the variable to be analyzed
 * @param mode : Choose to analyze read or write or both
 * @param includeTrivialBound : True to including `lower_i = 0` and `upper_i =
 * len_i - 1` as trivial bounds. False to return nullptr if no non-trivial bound
 * is found
 * @param filterSubTree : If set, consider only uses of the variable inside this
 * sub-tree
 */
AccessBound compAccessBound(const Stmt &op, const ID &varDefId,
                            CompAccessBoundMode mode = COMP_ACCESS_BOUND_ALL,
                            bool includeTrivialBound = true,
                            const ID &filterSubTree = ID());

} // namespace freetensor

#endif // FREE_TENSOR_COMP_ACCESS_BOUND_H
