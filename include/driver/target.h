#ifndef TARGET_H
#define TARGET_H

#include <string>

namespace ir {

enum class TargetType : int { CPU, GPU };

class Target {
  public:
    ~Target() = default;
    virtual TargetType type() const = 0;
    virtual std::string toString() const = 0;
};

class CPU : public Target {
  public:
    TargetType type() const override { return TargetType::CPU; }
    std::string toString() const override { return "CPU"; }
};

class GPU : public Target {
    int computeCapability_ = 0;

  public:
    TargetType type() const override { return TargetType::GPU; }
    std::string toString() const override { return "GPU"; }

    /// E.g. 70 for compute capability 7.0 (sm_70)
    void setComputeCapability(int value) { computeCapability_ = value; }
    int computeCapability() const { return computeCapability_; }
};

} // namespace ir

#endif // TARGET_H

