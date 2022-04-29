#include <driver/device.h>
#include <driver/gpu.h>

namespace freetensor {

void Device::sync() {
    switch (type()) {
    case TargetType::GPU:
        // FIXME: Switch to the specific device
        checkCudaError(cudaDeviceSynchronize());
    default:;
    }
}

} // namespace freetensor
