#include <driver/device.h>
#ifdef FT_WITH_CUDA
#include <driver/gpu.h>
#endif

namespace freetensor {

void Device::sync() {
    switch (type()) {
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        // FIXME: Switch to the specific device
        checkCudaError(cudaDeviceSynchronize());
#endif // FT_WITH_CUDA
    default:;
    }
}

} // namespace freetensor
