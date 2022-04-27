#ifndef FREE_TENSOR_FIND_INDEXING_LOOPS_H
#define FREE_TENSOR_FIND_INDEXING_LOOPS_H

#include <unordered_map>

#include <analyze/symbol_table.h>
#include <visitor.h>

namespace freetensor {

class FindIndexingLoops : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::unordered_map<For, std::vector<VarDef>> results_;
    std::vector<VarDef> inIndicesStack_;

  public:
    const std::unordered_map<For, std::vector<VarDef>> &results() const {
        return results_;
    }

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
};

/**
 * Find which which buffers is indexed by which loops
 */
inline std::unordered_map<For, std::vector<VarDef>>
findIndexingLoops(const Stmt &op) {
    FindIndexingLoops visitor;
    visitor(op);
    return visitor.results();
}

} // namespace freetensor

#endif // FREE_TENSOR_FIND_INDEXING_LOOPS_H
