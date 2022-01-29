#ifndef OUTPUT_INTERMEDIATES_H
#define OUTPUT_INTERMEDIATES_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace ir {

class OutputIntermediates : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<AST, Expr> &versions_;
    const std::unordered_map<std::string, Expr> &totLens_;
    std::unordered_map<std::string, std::string> tapeNames_;

  public:
    OutputIntermediates(const std::unordered_map<AST, Expr> &versions,
                        const std::unordered_map<std::string, Expr> &totLens)
        : versions_(versions), totLens_(totLens) {}

    const std::unordered_map<std::string, std::string> &tapeNames() const {
        return tapeNames_;
    }

  private:
    bool isSingleVersion(const std::string &defId) const;

  protected:
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * Save some specified intermediate (MemType::Cache) variables as outputs in a
 * program
 *
 * Old intermediate variables are still preserved, but may be removed using a
 * `inline` schedule. Rationale: one intermediate (old) element maps to multiple
 * output (new) elements, so it is hard to determine which element to load
 * from, if directly loading from the output variable
 *
 * @params op : The program
 * @params intermediates : VarDef IDs of the intermediate variables
 * @return : (
 *  The transformed program
 *  Mapping from VarDef IDs of intermediate variables to output names
 *  Versions of each memory accesses,
 *  Total version counts of each VarDef nodes
 * )
 */
std::tuple<Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<AST, Expr>, std::unordered_map<std::string, Expr>>
outputIntermediates(const Stmt &op,
                    const std::unordered_set<std::string> &intermediates);

} // namespace ir

#endif // OUTPUT_INTERMEDIATES_H
