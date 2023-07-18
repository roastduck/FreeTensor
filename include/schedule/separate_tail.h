#ifndef FREE_TENSOR_SEPERATE_TAIL_H
#define FREE_TENSOR_SEPERATE_TAIL_H

#include <functional>
#include <unordered_set>
#include <vector>

#include <analyze/symbol_table.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

class WrapMetadata : public Mutator {
    std::string op_;

  public:
    WrapMetadata(const std::string &op) : op_(op) {}

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

  public:
    SeparateTail(bool noDuplicateVarDefs,
                 const std::unordered_set<ID> &candidates)
        : noDuplicateVarDefs_(noDuplicateVarDefs), candidates_(candidates) {}

    const std::unordered_set<ID> &nextCandidates() const {
        return nextCandidates_;
    }

  private:
    void genSeparation(const Expr &iterVar, const Expr &cond,
                       const std::unordered_set<std::string> &bodyAllWrites,
                       const std::function<void(const Expr &)> &callback);

  protected:
    Stmt visit(const If &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt separateTail(const Stmt &ast, bool noDuplicateVarDefs);

} // namespace freetensor

#endif // FREE_TENSOR_SEPERATE_TAIL_H
