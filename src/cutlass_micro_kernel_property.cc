#include <cutlass_micro_kernel_property.h>
#include <hash.h>

namespace freetensor {

void CutlassMicroKernelProperty::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace freetensor
