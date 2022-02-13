#ifndef SEPERATE_TAIL_H
#define SEPERATE_TAIL_H

#include <functional>
#include <unordered_set>
#include <vector>

#include <analyze/analyze_linear.h>
#include <analyze/symbol_table.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllIfs : public Visitor {
    std::unordered_set<ID> results_;

  public:
    const std::unordered_set<ID> &results() const { return results_; }

  protected:
    void visit(const If &op) override;
};

class AppendIDs : public Mutator {
    std::string suffix_;

  public:
    AppendIDs(const std::string &suffix) : suffix_(suffix) {}

  protected:
    Stmt visitStmt(const Stmt &op) override;
};

/**
 * Separate main iterations and tail iterations of a loop
 *
 * E.g.
 *
 * ```
 * for i = 0 -> 3 {
 *   for j = 0 -> 4 {
 *      if (i * 4 + j < 10) {
 *        ...
 *      }
 *   }
 * }
 * ```
 *
 * Each loop will be separated into 2 parts: the body and the tail. After
 * simplification, the program will finally be transformed to
 *
 * ```
 * for i = 0 -> 2 {
 *   for j = 0 -> 4 {
 *     ...
 *   }
 * }
 * for j = 0 -> 2 {
 *   ...
 * }
 * ```
 */
class SeparateTail : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    bool noDuplicateVarDefs_;

    const std::unordered_set<ID> &candidates_;
    std::unordered_set<ID> nextCandidates_;

    std::vector<std::vector<If>> ifStack_;
    std::vector<bool> hasVarDefStack_;
    AnalyzeLinear analyzeLinear_;

  public:
    SeparateTail(bool noDuplicateVarDefs,
                 const std::unordered_set<ID> &candidates)
        : noDuplicateVarDefs_(noDuplicateVarDefs), candidates_(candidates) {}

    const std::unordered_set<ID> &nextCandidates() const {
        return nextCandidates_;
    }

  private:
    void genSeperation(const Expr &iterVar, const Expr &cond,
                       const std::function<void(const Expr &)> &callback);

  protected:
    Stmt visit(const If &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt separateTail(const Stmt &ast, bool noDuplicateVarDefs);

} // namespace ir

#endif // SEPERATE_TAIL_H
