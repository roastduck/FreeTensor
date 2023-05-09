#ifndef FREE_TENSOR_GPU_NORMALIZE_THREAD_DIMS_H
#define FREE_TENSOR_GPU_NORMALIZE_THREAD_DIMS_H

#ifdef FT_WITH_CUDA

#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class NormalizeThreadDims : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    CompUniqueBounds bound_;
    std::unordered_set<For> openLoopsInKernel_;
    bool inKernel_ = false;

  public:
    NormalizeThreadDims() : bound_(*this) {}

  private:
    /**
     * Ensure the length is defined with only constants and "byvalue" variables
     */
    bool isLegalLen(const Expr &expr);
    bool isLegalLen(const std::unordered_set<std::string> &names);

  protected:
    using BaseClass::visit;
    Stmt visit(const For &op) override;
};

/**
 * Express thread and block dimensions with variables out of kernels
 *
 * Semantics of the originl program will be preserved by adding conditions into
 * the kernel body
 */
inline Stmt normalizeThreadDims(const Stmt &ast) {
    return NormalizeThreadDims{}(ast);
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_NORMALIZE_THREAD_DIMS_H
