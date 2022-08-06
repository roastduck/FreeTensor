#ifndef FREE_TENSOR_TARGET_T_H
#define FREE_TENSOR_TARGET_T_H

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
class Target_t {
    bool useNativeArch_;
#ifdef FT_WITH_CUDA
    Ref<cudaDeviceProp> infoArch_;
#endif // FT_WITH_CUDA

  public:
    Target_t(bool useNativeArch = true) : useNativeArch_(useNativeArch) {}

    void setUseNativeArch(bool useNativeArch = true) {
        useNativeArch_ = useNativeArch;
    }
    bool useNativeArch() const { return useNativeArch_; }
    virtual ~Target_t() = default;
    virtual TargetType type() const = 0;
    virtual std::string toString() const = 0;
    virtual MemType mainMemType() const = 0;

#ifdef FT_WITH_CUDA
    void setInfoArch(const Ref<cudaDeviceProp> &infoArch = nullptr) {
        infoArch_ = infoArch;
    }
    const Ref<cudaDeviceProp> &infoArch() {
        if (!infoArch_.isValid()) {
            infoArch_ = Ref<cudaDeviceProp>::make();
        }
        return infoArch_;
    }

#endif // FT_WITH_CUDA
};

class CPU_t : public Target_t {
  public:
    CPU_t(bool useNativeArch = true) : Target_t(useNativeArch) {}

    TargetType type() const override { return TargetType::CPU; }
    std::string toString() const override { return "CPU"; }
    MemType mainMemType() const override { return MemType::CPU; }
};

class GPU_t : public Target_t {

  public:
    GPU_t(bool useNativeArch = true) : Target_t(useNativeArch) {}

    TargetType type() const override { return TargetType::GPU; }
    std::string toString() const override { return "GPU"; }
    MemType mainMemType() const override { return MemType::GPUGlobal; }
};

} // namespace freetensor

#endif // FREE_TENSOR_TARGET_T_H
