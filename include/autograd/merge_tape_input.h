#ifndef FREE_TENSOR_MERGE_TAPE_INPUT_H
#define FREE_TENSOR_MERGE_TAPE_INPUT_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

class MergeTapeInput : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<Stmt, std::vector<VarDef>> &lca2newNodes_;
    const std::unordered_set<std::string> &namesMerging_;

  public:
    MergeTapeInput(
        const std::unordered_map<Stmt, std::vector<VarDef>> &lca2newNodes,
        const std::unordered_set<std::string> &namesMerging)
        : lca2newNodes_(lca2newNodes), namesMerging_(namesMerging) {}

  protected:
    using BaseClass::visit;
    Stmt visitStmt(const Stmt &s) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * A backward program may re-input the same taped variable multiple times. We
 * need to merge these "input" VarDef nodes as one
 */
Stmt mergeTapeInput(const Stmt &op);

} // namespace freetensor

#endif // FREE_TENSOR_MERGE_TAPE_INPUT_H
