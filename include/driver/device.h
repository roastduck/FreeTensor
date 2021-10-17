#ifndef DEVICE_H
#define DEVICE_H

#include <driver/target.h>
#include <ref.h>

namespace ir {

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
};

} // namespace ir

#endif // DEVICE_H
