#ifndef FREE_TENSOR_TARGET_H
#define FREE_TENSOR_TARGET_H

#include <string>

#ifdef FT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <buffer.h>
#include <driver/target_type.h>
#include <opt.h>
#include <ref.h>

namespace freetensor {

/**
 * Target architecture
 */
class Target {

  public:
    Target() {}

    virtual ~Target() = default;
    virtual bool useNativeArch() const = 0;
    virtual TargetType type() const = 0;
    virtual std::string toString() const = 0;
    virtual MemType mainMemType() const = 0;
};

class CPU : public Target {
    bool useNativeArch_;
    // TODO: infoArch

  public:
    CPU(bool useNativeArch = true) : useNativeArch_(useNativeArch) {}

    void setUseNativeArch(bool useNativeArch = true) {
        useNativeArch_ = useNativeArch;
    }
    bool useNativeArch() const { return useNativeArch_; }
    TargetType type() const override { return TargetType::CPU; }
    std::string toString() const override { return "CPU"; }
    MemType mainMemType() const override { return MemType::CPU; }
};

#ifdef FT_WITH_CUDA
class GPU : public Target {
    Ref<cudaDeviceProp> infoArch_;

  public:
    // GPU is constructed from real local Deivce
    // `infoArch` has no default value
    // `useNativeArch` is always true
    GPU(const Ref<cudaDeviceProp> &infoArch) : infoArch_(infoArch) {}

    bool useNativeArch() const override { return true; }
    TargetType type() const override { return TargetType::GPU; }
    std::string toString() const override { return "GPU"; }
    MemType mainMemType() const override { return MemType::GPUGlobal; }

    void setInfoArch(const Ref<cudaDeviceProp> &infoArch) {
        infoArch_ = infoArch;
    }
    const Ref<cudaDeviceProp> &infoArch() const { return infoArch_; }

    Ref<std::pair<int, int>> computeCapability() const {
        return Ref<std::pair<int, int>>::make(
            std::make_pair(infoArch_->major, infoArch_->minor));
    }
};
#endif // FT_WITH_CUDA

bool isSameTarget(const Ref<Target> &lhs, const Ref<Target> &rhs);

} // namespace freetensor

#endif // FREE_TENSOR_TARGET_H
