#ifndef FREE_TENSOR_ALL_DEFS_H
#define FREE_TENSOR_ALL_DEFS_H

#include <unordered_set>

#include <visitor.h>

namespace freetensor {

class AllDefs : public Visitor {
    const std::unordered_set<AccessType> &atypes_;
    std::vector<std::pair<ID, std::string>> results_; // (id, name)

  public:
    AllDefs(const std::unordered_set<AccessType> &atypes) : atypes_(atypes) {}

    const std::vector<std::pair<ID, std::string>> &results() const {
        return results_;
    }

  protected:
    void visit(const VarDef &op) override;
};

/**
 * Collect IDs of all `VarDef` nodes of specific `AccessType`s
 */
std::vector<std::pair<ID, std::string>>
allDefs(const Stmt &op, const std::unordered_set<AccessType> &atypes = {
                            AccessType::Input, AccessType::Output,
                            AccessType::InOut, AccessType::Cache});

} // namespace freetensor

#endif // FREE_TENSOR_ALL_DEFS_H
