#ifndef SEPERATE_TAIL_H
#define SEPERATE_TAIL_H

#include <unordered_set>
#include <vector>

#include <analyze/linear.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllIfs : public Visitor {
    std::unordered_set<std::string> results_;

  public:
    const std::unordered_set<std::string> &results() const { return results_; }

  protected:
    void visit(const If &op) override;
};

class AppendIDs : public Mutator {
    std::string suffix_;

  public:
    AppendIDs(const std::string &suffix) : suffix_(suffix) {}

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
};

/**
 * Seperate main iterations and tail iterations of a loop
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
 * Each loop will be seperated into 2 parts: the body and the tail. After
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
class SeperateTail : public Mutator {
    const std::unordered_set<std::string> &candidates_;
    std::unordered_set<std::string> nextCandidates_;

    std::unordered_set<std::string> def_;
    std::vector<std::vector<If>> ifStack_;
    AnalyzeLinear analyzeLinear_;

  public:
    SeperateTail(const std::unordered_set<std::string> &candidates)
        : candidates_(candidates) {}

    const std::unordered_set<std::string> &nextCandidates() const {
        return nextCandidates_;
    }

  protected:
    Stmt visit(const If &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

class CountNestedFor : public Visitor {
    int curNested_ = 0, maxNested_ = 0;

  public:
    int maxNested() const { return maxNested_; }

  protected:
    void visit(const For &op) override;
};

Stmt seperateTail(const Stmt &op);

} // namespace ir

#endif // SEPERATE_TAIL_H
