#ifndef IR_MULTI_LEVEL_TILING_WITH_FUSION_H
#define IR_MULTI_LEVEL_TILING_WITH_FUSION_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/structs.h>

#include <array>
#include <random>

namespace ir {

class MultiLevelTilingWithFusionRule : public Rule {
    ElementWiseInfo toFuse_;
    std::string pat_;
    std::vector<int> fuseLevels_;
    MemType memType_;

  public:
    MultiLevelTilingWithFusionRule(TargetType target) {
        if (target == TargetType::CPU) {
            pat_ = "SSRSRS";
            fuseLevels_ = {1, 2};
            memType_ = MemType::CPU;
        }
        else {
            pat_ = "SSSRRSRS";
            fuseLevels_ = {3};
            memType_ = MemType::GPULocal;
        }
    }
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class MultiLevelTilingWithFusionPart : public MultiLevelTilingPart {
    MemType memType_;
    int level_;
    ElementWiseInfo toFuse_;

  public:
    void genRandAnnotation(std::default_random_engine &gen) override;
    explicit MultiLevelTilingWithFusionPart(ForsWithDataReuse fors,
                                            ElementWiseInfo toFuse, int level,
                                            std::string pat,
                                            MemType memType);
    void apply(Schedule &schedule) override;
    SketchPart mutate(std::default_random_engine &gen) override;
    SketchPart crossover(const SketchPart &part,
                         std::default_random_engine &gen) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;
    [[nodiscard]] size_t hash() const override;
    SketchPartType partType() override {
        return SketchPartType::MultiLevelTilingWithFusion;
    }
};

} // namespace ir

#endif // IR_MULTI_LEVEL_TILING_WITH_FUSION_H
