#ifndef FREE_TENSOR_TARGET_H
#define FREE_TENSOR_TARGET_H

#include <string>

#include <buffer.h>
#include <ref.h>

namespace freetensor {

enum class TargetType : int { CPU, GPU };

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
  public:
    CPU(bool useNativeArch = true) : Target(useNativeArch) {}

    TargetType type() const override { return TargetType::CPU; }
    std::string toString() const override { return "CPU"; }
    MemType mainMemType() const override { return MemType::CPU; }
};

class GPU : public Target {
    Ref<std::pair<int, int>> computeCapability_;

  public:
    GPU(bool useNativeArch = true) : Target(useNativeArch) {}

    TargetType type() const override { return TargetType::GPU; }
    std::string toString() const override { return "GPU"; }
    MemType mainMemType() const override { return MemType::GPUGlobal; }

    /// E.g. (7, 0) for compute capability 7.0 (sm_70)
    void setComputeCapability(int major, int minor) {
        computeCapability_ =
            Ref<std::pair<int, int>>::make(std::make_pair(major, minor));
    }
    Ref<std::pair<int, int>> computeCapability() const {
        return computeCapability_;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_TARGET_H
