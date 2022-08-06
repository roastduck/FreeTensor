#ifndef FREE_TENSOR_DEVICE_T_H
#define FREE_TENSOR_DEVICE_T_H

#include <driver/target_t.h>
#include <ref.h>

namespace freetensor {

/**
 * A computing device constructed from
 *      1. (TargetType, DeviceNumber)
 *
 * E.g. Device(TargetType::GPU, 0) means the 0-th GPU (device)
 */

class Device_t {
    TargetType targetType_;
    size_t num_;
    Ref<Target_t> target_;

  public:
    Device_t(const TargetType &targetType, size_t num = 0)
        : targetType_(targetType), num_(num) {}
    Device_t(const TargetType &targetType, const std::string &getDeviceByName);

    TargetType type() const { return targetType_; }
    size_t num() const { return num_; }
    const Ref<Target_t> &target();
};

} // namespace freetensor

#endif // FREE_TENSOR_DEVICE_T_H
