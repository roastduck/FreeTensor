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
    bool useNativeArch_;

  public:
    Target(bool useNativeArch = true) : useNativeArch_(useNativeArch) {}

    void setUseNativeArch(bool useNativeArch = true) {
        useNativeArch_ = useNativeArch;
    }
    bool useNativeArch() const { return useNativeArch_; }
    virtual ~Target() = default;
    virtual TargetType type() const = 0;
    virtual std::string toString() const = 0;
    virtual MemType mainMemType() const = 0;
};

class CPU : public Target {
    // TODO: infoArch

  public:
    CPU(bool useNativeArch = true) : Target(useNativeArch) {}

    TargetType type() const override { return TargetType::CPU; }
    std::string toString() const override { return "CPU"; }
    MemType mainMemType() const override { return MemType::CPU; }
};

class GPU : public Target {
#ifdef FT_WITH_CUDA
    Ref<cudaDeviceProp> infoArch_;
#endif // FT_WITH_CUDA

  public:
#ifndef FT_WITH_CUDA
    GPU(bool useNativeArch = true) : Target(useNativeArch) {}
#endif // NOT FT_WITH_CUDA

#ifdef FT_WITH_CUDA
    GPU(const Ref<cudaDeviceProp> &infoArch = nullptr,
        bool useNativeArch = true)
        : Target(useNativeArch), infoArch_(infoArch) {}
#endif // FT_WITH_CUDA

    TargetType type() const override { return TargetType::GPU; }
    std::string toString() const override { return "GPU"; }
    MemType mainMemType() const override { return MemType::GPUGlobal; }

#ifdef FT_WITH_CUDA
    void setInfoArch(const Ref<cudaDeviceProp> &infoArch = nullptr) {
        infoArch_ = infoArch;
    }
    const Ref<cudaDeviceProp> &infoArch() const { return infoArch_; }

    Ref<std::pair<int, int>> computeCapability() const {
        if (!infoArch_.isValid())
            return nullptr;
        return Ref<std::pair<int, int>>::make(
            std::make_pair(infoArch_->major, infoArch_->minor));
    }
#endif // FT_WITH_CUDA
};

bool isSameTarget(const Ref<Target> &lhs, const Ref<Target> &rhs);

} // namespace freetensor

#endif // FREE_TENSOR_TARGET_H
