#ifndef ANALYZE_VERSION_H
#define ANALYZE_VERSION_H

#include <unordered_map>
#include <unordered_set>

#include <visitor.h>

namespace ir {

class CountScopeLen : public Visitor {
    std::string def_, var_;
    const std::unordered_set<std::string> &affectingScopes_; // For IDs
    const std::unordered_set<std::string> &needTapes_; // Store or ReduceTo IDs
    std::unordered_map<Stmt, Expr> scopeLen_;

  public:
    CountScopeLen(const std::string &def,
                  const std::unordered_set<std::string> &affectingScopes,
                  const std::unordered_set<std::string> &needTapes)
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

class AnalyzeVersion : public Visitor {
    std::string def_, var_;
    const std::unordered_set<std::string> &affectingScopes_; // For IDs
    const std::unordered_set<std::string> &needTapes_; // Store or ReduceTo IDs
    const std::unordered_map<Stmt, Expr> &scopeLen_;
    Expr totLen_;
    std::unordered_map<AST, Expr> &versions_;
    std::string tapeName_;
    Expr offset_ = makeIntConst(0);

  public:
    AnalyzeVersion(const std::string &def,
                   const std::unordered_set<std::string> &affectingScopes,
                   const std::unordered_set<std::string> &needTapes,
                   const std::unordered_map<Stmt, Expr> &scopeLen,
                   const Expr &totLen, std::unordered_map<AST, Expr> &versions)
        : def_(def), affectingScopes_(affectingScopes), needTapes_(needTapes),
          scopeLen_(scopeLen), totLen_(totLen), versions_(versions) {}

    const std::string &tapeName() const { return tapeName_; }

  protected:
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

/**
 * Assign each memory access an expression that identifies each version of the
 * accessed variable
 *
 * @return : (node -> versions, VarDef IDs -> total version counts)
 */
std::pair<std::unordered_map<AST, Expr>, std::unordered_map<std::string, Expr>>
analyzeVersion(const Stmt &op,
               const std::unordered_set<std::string> &intermediates);

} // namespace ir

#endif // ANALYZE_VERSION_H
