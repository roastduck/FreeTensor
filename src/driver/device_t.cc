#include <cstring>
#include <driver/device_t.h>
#include <driver/gpu.h>
namespace freetensor {

Device_t::Device_t(const TargetType &targetType,
                   const std::string &getDeviceByName)
    : targetType_(targetType) {
    switch (targetType) {
    case TargetType::CPU: {

        break;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        auto deviceModel = Ref<cudaDeviceProp>::make();
        memset(&(*deviceModel), 0, sizeof(cudaDeviceProp));
        strncpy(deviceModel->name, getDeviceByName.c_str(), 255);
        checkCudaError(cudaChooseDevice((int *)&num_, &(*deviceModel)));
        break;
    }
#endif // FT_WITH_CUDA
    default:;
    }
}

const Ref<Target_t> &Device_t::target() {
    switch (type()) {
    case TargetType::CPU: {
        ASSERT(false);
        if (!target_.isValid()) {
        }
        break;
    }
#ifdef FT_WITH_CUDA
    case TargetType::GPU: {
        if (!target_.isValid()) {
            target_ = Ref<GPU_t>::make();
            checkCudaError(cudaGetDeviceProperties(&(*(target_->infoArch())), num_));
        }
        break;
    }
#endif // FT_WITH_CUDA
    default:;
    }
    return target_;
}

} // namespace freetensor
