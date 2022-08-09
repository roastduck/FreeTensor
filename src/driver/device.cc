#include <cstring>
#include <driver/device.h>
#include <driver/gpu.h>
namespace freetensor {

static const size_t deviceNameMaxLength =
    255; // not 256, for '\0' terminated string

Device::Device(const TargetType &targetType, int num) : num_(num) {
    switch (targetType) {
    case TargetType::CPU: {
        // TODO: infoArch
        target_ = Ref<CPUTarget>::make();
        break;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto deviceModel = Ref<cudaDeviceProp>::make();
        checkCudaError(cudaGetDeviceProperties(&(*deviceModel), num_));
        target_ = Ref<GPUTarget>::make(deviceModel);
        break;
    }
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}
Device::Device(const TargetType &targetType,
               const std::string &getDeviceByName) {
    switch (targetType) {
    case TargetType::CPU: {
        // TODO
        ASSERT(false);

        break;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {

        /* fetch a device best matches deviceModel */
        auto deviceModel = Ref<cudaDeviceProp>::make();
        memset(&(*deviceModel), 0, sizeof(cudaDeviceProp));
        strncpy(deviceModel->name, getDeviceByName.c_str(),
                deviceNameMaxLength);
        checkCudaError(cudaChooseDevice(&num_, &(*deviceModel)));

        /* get the device architecture info */
        checkCudaError(cudaGetDeviceProperties(&(*deviceModel), num_));
        target_ = Ref<GPUTarget>::make(deviceModel);
        break;
    }
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

Device::Device(const TargetType &targetType,
               const std::string &getDeviceByFullName, size_t nth) {
    switch (targetType) {
    case TargetType::CPU: {
        // TODO
        ASSERT(false);
        break;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {

        int deviceCount;
        checkCudaError(cudaGetDeviceCount(&deviceCount));
        auto deviceModel = Ref<cudaDeviceProp>::make();
        num_ = -1;
        for (int i = 0; i < deviceCount; i++) {
            checkCudaError(cudaGetDeviceProperties(&(*deviceModel), i));
            if (strncmp(deviceModel->name, getDeviceByFullName.c_str(),
                        deviceNameMaxLength) == 0) {
                if (!nth) {
                    num_ = i;
                    break;
                }
                nth--;
            }
        }
        ASSERT(num_ != -1);
        checkCudaError(cudaGetDeviceProperties(&(*deviceModel), num_));
        target_ = Ref<GPUTarget>::make(deviceModel);
        break;
    }
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

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
