#ifndef FREE_TENSOR_CACHE_WRITE_H
#define FREE_TENSOR_CACHE_WRITE_H
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>

namespace freetensor {

class CacheWriteRule : public Rule {
    MemType memType_;

  public:
    explicit CacheWriteRule(TargetType target)
        : memType_(target == TargetType::CPU ? MemType::CPU
                                             : MemType::GPULocal) {}
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

} // namespace freetensor
#endif // FREE_TENSOR_CACHE_WRITE_H
