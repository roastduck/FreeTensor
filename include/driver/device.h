#ifndef FREE_TENSOR_DEVICE_H
#define FREE_TENSOR_DEVICE_H

#include <driver/target.h>
#include <ref.h>

namespace freetensor {

class Device {
    Ref<Target> target_;
    size_t num_;

  public:
    Device(const Ref<Target> &target, size_t num = 0)
        : target_(target), num_(num) {}

    const Ref<Target> &target() const { return target_; }
    size_t num() const { return num_; }

    TargetType type() const { return target_->type(); }
    MemType mainMemType() const { return target_->mainMemType(); }

    void sync();
};

} // namespace freetensor

#endif // FREE_TENSOR_DEVICE_H
