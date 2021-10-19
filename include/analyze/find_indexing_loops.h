#ifndef FIND_INDEXING_LOOPS_H
#define FIND_INDEXING_LOOPS_H

#include <unordered_map>

#include <visitor.h>

namespace ir {

class FindIndexingLoops : public Visitor {
    std::unordered_map<For, std::vector<VarDef>> results_;
    std::unordered_map<std::string, VarDef> defs_; // name -> def
    std::unordered_map<std::string, For> loops_;   // name -> for
    std::vector<VarDef> inIndicesStack_;

  public:
    const std::unordered_map<For, std::vector<VarDef>> &results() const {
        return results_;
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
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

} // namespace ir

#endif // FIND_INDEXING_LOOPS_H
