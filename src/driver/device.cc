#include <driver/device.h>
#include <driver/gpu.h>

namespace ir {

void Device::sync() {
    switch (type()) {
    case TargetType::GPU:
        // FIXME: Switch to the specific device
        checkCudaError(cudaDeviceSynchronize());
    default:;
    }
}

} // namespace ir
