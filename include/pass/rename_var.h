#ifndef FREE_TENSOR_RENAME_VAR_H
#define FREE_TENSOR_RENAME_VAR_H

#include <unordered_map>

#include <mutator.h>

namespace freetensor {

class RenameVar : public Mutator {
  protected:
    std::unordered_map<std::string, std::string> rename_;

  public:
    RenameVar() {}
    RenameVar(const std::unordered_map<std::string, std::string> &rename)
        : rename_(rename) {}

  protected:
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const Alloc &op) override;
    Stmt visit(const Free &op) override;
    Stmt visit(const MarkVersion &op) override;
};

/**
 * Rename a variable's definition and use sites
 *
 * This function can be applied to an AST sub-tree
 *
 * @{
 */
inline Stmt
renameVar(const Stmt &op,
          const std::unordered_map<std::string, std::string> &rename) {
    return RenameVar(rename)(op);
}
inline Stmt renameVar(const Stmt &op, const std::string &oldName,
                      const std::string &newName) {
    return renameVar(op, {{oldName, newName}});
}
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_RENAME_VAR_H
