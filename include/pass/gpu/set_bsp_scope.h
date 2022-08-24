#ifndef FREE_TENSOR_GPU_SET_BSP_SCOPE_H
#define FREE_TENSOR_GPU_SET_BSP_SCOPE_H

#include <func.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class SetBSPScope : public Mutator {
    bool inKernel_ = false;

  protected:
    Stmt visit(const BSPScope &op) override;
    Stmt visit(const For &op) override;
};

/**
 * Ensure each GPU thread is enclosed in a BSPScope, which represents a kernel
 *
 * If there is no BSPScope, create one for each outer-most threadIdx or blockIdx
 * For node
 */
inline Stmt setBSPScope(const Stmt &op) { return SetBSPScope{}(op); }

DEFINE_PASS_FOR_FUNC(setBSPScope)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_SET_BSP_SCOPE_H
