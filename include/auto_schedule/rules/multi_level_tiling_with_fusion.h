#ifndef FREE_TENSOR_MULTI_LEVEL_TILING_WITH_FUSION_H
#define FREE_TENSOR_MULTI_LEVEL_TILING_WITH_FUSION_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/structs.h>

#include <array>
#include <random>

namespace freetensor {

class MultiLevelTilingWithFusionRule : public Rule {
    ElementWiseInfo toFuse_;
    std::string pat_;
    std::vector<int> fuseLevels_;
    TargetType targetType_;

  public:
    MultiLevelTilingWithFusionRule(TargetType target) : targetType_(target) {
        if (target == TargetType::CPU) {
            pat_ = "SSRSRS";
            fuseLevels_ = {1, 2};
        } else {
            pat_ = "SSSRRSRS";
            fuseLevels_ = {3};
        }
    }
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class MultiLevelTilingWithFusionPart : public MultiLevelTilingPart {
    TargetType targetType_;
    int level_;
    ElementWiseInfo toFuse_;

  public:
    void genRandAnnotation(std::default_random_engine &gen) override;
    explicit MultiLevelTilingWithFusionPart(ForsWithDataReuse fors,
                                            ElementWiseInfo toFuse, int level,
                                            std::string pat,
                                            TargetType targetType);
    void apply(Schedule &schedule, SketchTarget &target) override;
    bool mutate(std::default_random_engine &gen) override;
    bool crossover(const SketchPart &part,
                   std::default_random_engine &gen) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;
    [[nodiscard]] size_t hash() const override;
    SketchPartType partType() override {
        return SketchPartType::MultiLevelTilingWithFusion;
    }
    [[nodiscard]] SketchPart clone() const override {
        return Ref<MultiLevelTilingWithFusionPart>::make(*this);
    }
    void printAnnotation() {
        std::cout << "/*space: */{\n";
        for (const auto &anno : annotation_.spaceLoopTiling) {
            std::cout << "\t{";
            for (const auto &i : anno) {
                std::cout << i << ", ";
            }
            std::cout << "},\n";
        }
        std::cout << "},\n/*reduction: */{\n";
        for (const auto &anno : annotation_.reductionLoopTiling) {
            std::cout << "\t{";
            for (const auto &i : anno) {
                std::cout << i << ", ";
            }
            std::cout << "},\n";
        }
        std::cout << "}" << std::endl;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_MULTI_LEVEL_TILING_WITH_FUSION_H
