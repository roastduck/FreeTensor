#ifndef FREE_TENSOR_GPU_NORMALIZE_VAR_IN_KERNEL_H
#define FREE_TENSOR_GPU_NORMALIZE_VAR_IN_KERNEL_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class NormalizeVarInKernel : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    std::vector<std::string> legalNames_;

    std::vector<VarDef> varsToHoist_;
    std::unordered_set<std::string> usedNamesInKernel_;
    std::unordered_map<std::string, int> nameCntInKernel_;
    bool inKernel_ = false;

    CompUniqueBounds unique_;

  public:
    NormalizeVarInKernel() : unique_(*this) {}

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
};

/**
 * The shape of all variables defined inside a kernel must be determined outside
 * the kernel
 */
Stmt normalizeVarInKernel(const Stmt &s);

DEFINE_PASS_FOR_FUNC(normalizeVarInKernel)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_NORMALIZE_VAR_IN_KERNEL_H
