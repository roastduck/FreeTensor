#ifndef FREE_TENSOR_ANALYZE_VERSION_H
#define FREE_TENSOR_ANALYZE_VERSION_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <autograd/derivative.h>
#include <visitor.h>

namespace freetensor {

class CountScopeLen : public Visitor {
    ID def_;
    std::string var_;
    const std::unordered_set<ID> &affectingScopes_; // For IDs
    const std::unordered_set<ID> &needTapes_;       // Store or ReduceTo IDs
    std::unordered_map<Stmt, Expr> scopeLen_;

  public:
    CountScopeLen(const ID &def, const std::unordered_set<ID> &affectingScopes,
                  const std::unordered_set<ID> &needTapes)
        : def_(def), affectingScopes_(affectingScopes), needTapes_(needTapes) {}

    const std::unordered_map<Stmt, Expr> &scopeLen() const { return scopeLen_; }

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
};

class AnalyzeVersion : public TrackStmt<Visitor> {
    typedef TrackStmt<Visitor> BaseClass;

    ID def_;
    std::string var_;
    const std::unordered_set<ID> &affectingScopes_; // For IDs
    const std::unordered_set<ID> &needTapes_;       // Store or ReduceTo IDs
    const std::unordered_map<Stmt, Expr> &scopeLen_;
    Expr totLen_;
    std::unordered_map<StmtOrExprID, Expr> &versions_;
    std::unordered_map<std::string, std::pair<std::string, Expr>>
        &userVersions_;
    std::string tapeName_;
    Expr offset_ = makeIntConst(0);

  public:
    AnalyzeVersion(const ID &def, const std::unordered_set<ID> &affectingScopes,
                   const std::unordered_set<ID> &needTapes,
                   const std::unordered_map<Stmt, Expr> &scopeLen,
                   const Expr &totLen,
                   std::unordered_map<StmtOrExprID, Expr> &versions,
                   std::unordered_map<std::string, std::pair<std::string, Expr>>
                       &userVersions)
        : def_(def), affectingScopes_(affectingScopes), needTapes_(needTapes),
          scopeLen_(scopeLen), totLen_(totLen), versions_(versions),
          userVersions_(userVersions) {}

    const std::string &tapeName() const { return tapeName_; }

  protected:
    void visit(const Load &op) override;
    void visit(const MarkVersion &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

/**
 * Special for input variables. We won't call `AnalyzeVersion` on input
 * variables, but we still need them in `userVersions_`, so we assign them here
 */
class SetUserVersionsForInputs : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::unordered_map<std::string, std::pair<std::string, Expr>>
        &userVersions_;

  public:
    SetUserVersionsForInputs(
        std::unordered_map<std::string, std::pair<std::string, Expr>>
            &userVersions)
        : userVersions_(userVersions) {}

  protected:
    using BaseClass::visit;
    void visit(const MarkVersion &op) override;
};

/**
 * Assign each memory access an expression that identifies each version of the
 * accessed variable
 *
 * Versions are guaranteed to distinguish READ sites that may reads different
 * values. Versions of WRITE sites are assigned to be consistent to the READ
 * sites. This also means the following:
 *
 * - Multiple READ sites may have the same version
 * - There can be multiple WRITE sites even there is only one version
 *
 * Some variables are TRIVIAL and there is no need to distinguish their
 * versions. This function also outputs information of tribial variables. A
 * variable is considered trivial if:
 *
 * - There is only one version, AND
 * - The value of the only version is not overwritten till the end of the
 * variable's lifetime
 *
 * @param op : The AST to analyze
 * @param intermediates : Varaibles (VarDef IDs) to analyze
 * @param derivatives : Lazy derivative generators for each statement. We need
 * this information to determine which values are to be used in gradients
 * @param localVersionsOnly : If true, analyze local versions inside its VarDef
 * node. If false, analyze global versions within the whole program
 * @return : (node -> versions, VarDef IDs -> total version counts, trivial
 * VarDef IDs, tape_name -> var name, explicit user versions marked via
 * mark_version)
 */
std::tuple<std::unordered_map<StmtOrExprID, Expr>, std::unordered_map<ID, Expr>,
           std::unordered_set<ID>,
           std::unordered_map<std::string, std::pair<std::string, Expr>>>
analyzeVersion(
    const Stmt &op, const std::unordered_set<ID> &intermediates,
    const std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives,
    bool localVersionsOnly);

} // namespace freetensor

#endif // FREE_TENSOR_ANALYZE_VERSION_H
