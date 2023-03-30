#ifndef REFINE_SIGN_DATA_TYPE_H
#define REFINE_SIGN_DATA_TYPE_H

#include <unordered_map>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

/**
 * Set all non-I/O VarDef nodes to Never type.
 */
class ClearDataType : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::unordered_map<ID, DataType> oldTypes_; // {VarDef ID -> type}

  public:
    const auto &oldTypes() const { return oldTypes_; }

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
};

/**
 * Apply new data type to Load nodes and viewers
 */
class UpdateDType : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
};

/**
 * Check for each Store or ReduceTo. Type of a VarDef node becomes the union
 * with type of the expressions writing to it
 */
class RefineSignDataType : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, DataType> &userTypes_; // {VarDef ID -> type}
    std::unordered_map<ID, DataType> newTypes_;         // {VarDef ID -> type}
    bool converged_ = true;

  public:
    RefineSignDataType(const std::unordered_map<ID, DataType> &userTypes)
        : userTypes_(userTypes) {}

    bool converged() const { return converged_; }

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Try to set VarDef nodes to use more restricted SignDataType
 *
 * This pass only impacts SignDataType and not BaseDataType. In theory, the same
 * algorithm could be applied to BaseDataType as well. However, users tend to
 * explicitly set BaseDataType for implicit casting. For example, `y += x[i]` is
 * often done with a short `x` and a longer `y` for floating-point precision,
 * which is actually an implicit up-cast on `x`. If we were to apply this pass
 * on BaseDataType, we would cancel this cast.
 *
 * Algorithm:
 *
 * 1. Set all non-I/O VarDef nodes to Never type.
 * 2. Check for each Store or ReduceTo. Type of a VarDef node becomes the union
 * with type of the expressions writing to it. Propagate until converged.
 */
Stmt refineSignDataType(const Stmt &op);

DEFINE_PASS_FOR_FUNC(refineSignDataType);

} // namespace freetensor

#endif // REFINE_SIGN_DATA_TYPE_H
