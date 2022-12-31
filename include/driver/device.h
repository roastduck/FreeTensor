#ifndef FREE_TENSOR_DEVICE_H
#define FREE_TENSOR_DEVICE_H

#include <iostream>

#include <driver/target.h>
#include <ref.h>

namespace freetensor {

/**
 * A computing device can be constructed from
 *      1. (TargetType, DeviceNumber)
 *      2. (TargetType, getDeviceByName): cuda uses best matches criteria.
 *      3. (TargetType, FullName, nth): get nth(from 0) device named `Fullname`.
 *
 * E.g. Device(TargetType::GPU, 0) means the 0-th GPU (device)
 *      Device(TargetType::GPU, "V100") means a GPU which best matches "V100"
 *      Device(TargetType::GPU, "NVIDIA GeForce RTX 3060 Laptop GPU", 0)
 */

class Device {
    Ref<Target> target_;
    int num_; // not size_t, cuda function takes ints as args

  public:
    Device(const TargetType &targetType, int num = 0);
    Device(const TargetType &targetType, const std::string &getDeviceByName);
    Device(const TargetType &targetType, const std::string &getDeviceByFullName,
           size_t nth);

    TargetType type() const { return target_->type(); }
    MemType mainMemType() const { return target_->mainMemType(); }
    int num() const { return num_; }
    const Ref<Target> &target() const { return target_; }

    void sync();

    friend bool operator==(const Device &lhs, const Device &rhs) {
        return isSameTarget(lhs.target_, rhs.target_) && lhs.num_ == rhs.num_;
    }

    friend std::ostream &operator<<(std::ostream &os, const Device &device) {
        return os << device.target_ << ':' << device.num_;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_DEVICE_H
